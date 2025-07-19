# Copyright 2022 CircuitNet. All rights reserved.
import os
import os.path as osp
import cv2
import numpy as np
import torch
import multiprocessing as mul
import uuid
import psutil
import time
import csv
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, r2_score, roc_auc_score
from scipy.interpolate import make_interp_spline
from scipy.stats import pearsonr, spearmanr, kendalltau
from functools import partial, wraps
from inspect import getfullargspec
from mmcv import scandir
from scipy.stats import wasserstein_distance
from skimage.metrics import normalized_root_mse, structural_similarity as ssim_func
import math

__all__ = ['psnr', 'ssim', 'nrms', 'emd', 'peak_nrms', 'mae', 'r2', 'pearson', 'spearman', 'kendall', 'auc',
           'NRMS', 'SSIM', 'EMD', 'MAE', 'PEAK_NRMS', 'R2', 'PEARSON', 'SPEARMAN', 'KENDALL', 'AUC']

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '': return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

def input_converter(apply_to=None, out_type=np.float32):
    def input_converter_wrapper(old_func):
        @wraps(old_func)
        def new_func(*args, **kwargs):
            args_info = getfullargspec(old_func)
            args_to_cast = args_info.args if apply_to is None else apply_to
            new_args = []
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        # 对于回归任务，使用float32保持原始数据范围
                        new_args.append(tensor2img(args[i], out_type=out_type))
                    else:
                        new_args.append(args[i])
            # 确保kwargs也被传递
            return old_func(*new_args, **kwargs)
        return new_func
    return input_converter_wrapper

@input_converter(apply_to=('img1', 'img2'), out_type=np.uint8)
def psnr(img1, img2, crop_border=0):
    assert img1.shape == img2.shape, f'Image shapes differ: {img1.shape}, {img2.shape}'
    img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]
    mse_value = np.mean((img1 - img2)**2)
    return float('inf') if mse_value == 0 else 20. * np.log10(255. / np.sqrt(mse_value))

def _ssim(img1, img2):
    # 检查非法数值
    if np.isnan(img1).any() or np.isnan(img2).any():
        return np.nan
    if np.isinf(img1).any() or np.isinf(img2).any():
        return np.nan

    # 初始化常数
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # 转换数据类型
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # 生成高斯核
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    # 卷积操作
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # 裁剪边缘
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    # 处理卷积后空数组
    if mu1.size == 0 or mu2.size == 0:
        return 0.0  # 返回最低相似度

    # 计算方差协方差
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    # 数值稳定性处理
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    denominator = np.clip(denominator, 1e-8, None)  # 避免除零

    # 计算SSIM映射
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / denominator
    ssim_map = np.clip(ssim_map, -1.0, 1.0)  # 约束理论范围

    return np.nanmean(ssim_map)  # 自动忽略残留NaN


@input_converter(apply_to=('img1', 'img2'), out_type=np.uint8)
def ssim(img1, img2, crop_border=0):
    # 输入校验
    assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"

    # 动态调整裁剪边界
    h, w = img1.shape[0], img1.shape[1]
    max_crop = min(h, w) // 2 - 1
    crop_border = min(crop_border, max_crop) if max_crop > 0 else 0

    # 执行裁剪
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, :]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, :]
        h, w = img1.shape[0], img1.shape[1]

        # 校验有效尺寸
        if h < 11 or w < 11:
            raise ValueError(
                f"Invalid cropped size {h}x{w}. Minimum required: 11x11 after cropping {crop_border} pixels"
            )

    # 分通道计算
    valid_ssims = []
    for c in range(img1.shape[2]):
        channel_ssim = _ssim(img1[..., c], img2[..., c])
        if np.isfinite(channel_ssim):  # 过滤无效结果
            valid_ssims.append(channel_ssim)

    # 处理全无效通道情况
    if not valid_ssims:
        return 0.0  # 或根据需求返回np.nan

    return np.mean(valid_ssims)
# def _ssim(img1, img2):
#     C1 = (0.01 * 255)**2
#     C2 = (0.03 * 255)**2
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())
#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1**2
#     mu2_sq = mu2**2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()
#
# @input_converter(apply_to=('img1', 'img2'))
# def ssim(img1, img2, crop_border=0):
#     assert img1.shape == img2.shape, f'Image shapes differ: {img1.shape}, {img2.shape}'
#     if crop_border != 0:
#         img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
#         img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]
#     ssims = [_ssim(img1[..., i], img2[..., i]) for i in range(img1.shape[2])]
#     return np.array(ssims).mean()

@input_converter(apply_to=('img1', 'img2'), out_type=np.float32)
def nrms(img1, img2, crop_border=0):
    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    nrmse_value = normalized_root_mse(img1.flatten(), img2.flatten(),normalization='min-max')
    if math.isinf(nrmse_value):
        return 0.05
    return nrmse_value


@input_converter(apply_to=('img1', 'img2'), out_type=np.float32)
def mae(img1, img2, crop_border=0):
    """
    Calculate Mean Absolute Error (MAE) between two images.

    Args:
        img1 (np.ndarray): Ground truth image.
        img2 (np.ndarray): Predicted image.
        crop_border (int): Number of pixels to crop from borders (default: 0).

    Returns:
        float: MAE value.
    """
    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    mae_value = np.mean(np.abs(img1 - img2))
    return mae_value

#
# def peak_nrms(img1, img2, percentages=[0.5, 1, 2, 5], crop_border=0):
#     """
#     Calculate Peak NRMSE for top k% regions based on target values.
#
#     Args:
#         img1 (np.ndarray): Ground truth image (target).
#         img2 (np.ndarray): Predicted image.
#         percentages (list): List of top percentages to evaluate (default: [0.5, 1, 2, 5]).
#         crop_border (int): Number of pixels to crop from borders (default: 0).
#
#     Returns:
#         tuple: (peak_nrmse_avg, nrmse_dict)
#             - peak_nrmse_avg (float): Average NRMSE across all percentages.
#             - nrmse_dict (dict): NRMSE values for each percentage.
#     """
#     assert img1.shape == img2.shape, (
#         f'Image shapes do not match: {img1.shape} vs {img2.shape}')
#
#     # Crop borders if specified
#     if crop_border != 0:
#         img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
#         img2 = img2[crop_border:-crop_border, crop_border:-crop_border]
#
#     # Flatten images to 1D arrays
#     pred_flat = img2.flatten()  # img2 is prediction
#     target_flat = img1.flatten()  # img1 is ground truth
#     total_elements = len(target_flat)
#
#     # Sort target values in descending order to get top k% indices
#     sorted_indices = np.argsort(target_flat)[::-1]
#     nrmse_values = {}
#
#     # Calculate NRMSE for each percentage
#     for p in percentages:
#         num_elements = max(1, int(total_elements * p / 100))  # Ensure at least 1 element
#         selected_pred = pred_flat[sorted_indices[:num_elements]]
#         selected_target = target_flat[sorted_indices[:num_elements]]
#
#         # Compute NRMSE with dynamic denominator (min-max normalization)
#         nrmse = normalized_root_mse(selected_target, selected_pred, normalization='min-max')
#
#         # Handle inf/nan cases
#         if math.isinf(nrmse) or math.isnan(nrmse):
#             nrmse = 0.05  # Default fallback value
#         nrmse_values[p] = nrmse
#
#     # Compute average Peak NRMSE
#     peak_nrmse_avg = np.mean(list(nrmse_values.values()))
#
#     return peak_nrmse_avg, nrmse_values
@input_converter(apply_to=('img1', 'img2'))
def emd(img1, img2, crop_border=0):
    assert img1.shape == img2.shape, f'Image shapes differ: {img1.shape}, {img2.shape}'
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]
    img1 = normalize_exposure(np.squeeze(img1, axis=2))
    img2 = normalize_exposure(np.squeeze(img2, axis=2))
    hist_1 = get_histogram(img1)
    hist_2 = get_histogram(img2)
    return wasserstein_distance(hist_1, hist_2)

# 添加大写别名以便兼容
NRMS = nrms
SSIM = ssim
EMD = emd
MAE = mae

@input_converter(apply_to=('img1', 'img2'), out_type=np.float32)
def peak_nrms(img1, img2, crop_border=0, top_percentiles=[0.5, 1, 2, 5], return_dict=False):
    """
    计算Peak NRMSE指标，评估高拥塞区域的预测精度

    Args:
        img1 (np.ndarray): 真实值图像 (ground truth)
        img2 (np.ndarray): 预测值图像 (prediction)
        crop_border (int): 边界裁剪像素数
        top_percentiles (list): 要评估的高值百分位数列表
        return_dict (bool): 是否返回详细字典，默认返回平均值

    Returns:
        float or dict: 平均Peak NRMSE值或详细字典
    """
    assert img1.shape == img2.shape, f'Image shapes differ: {img1.shape}, {img2.shape}'
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    # img1是ground truth, img2是prediction
    flat_target = img1.flatten()  # 真实值
    flat_pred = img2.flatten()    # 预测值
    n = len(flat_target)

    peak_nrms_values = {}
    for p in top_percentiles:
        k = int(n * p / 100)
        if k <= 0:  # 确保至少选择一个点
            k = 1

        # 获取真实值图像中值最大的k个像素的索引
        top_indices = np.argpartition(flat_target, -k)[-k:]
        peak_target = flat_target[top_indices]
        peak_pred = flat_pred[top_indices]

        # 计算MSE
        mse = np.mean((peak_pred - peak_target) ** 2)

        # 安全计算分母 - 使用真实值的范围进行归一化
        img_range = np.max(peak_target) - np.min(peak_target)

        # 如果范围太小，使用默认值
        if img_range < 1e-8:
            peak_nrms_values[p] = 0.05
            continue

        # 计算NRMSE
        nrmse = np.sqrt(mse) / img_range

        # 处理无效值
        peak_nrms_values[p] = 0.05 if (math.isinf(nrmse) or math.isnan(nrmse)) else nrmse

    # 返回字典或平均值
    if return_dict:
        return peak_nrms_values
    else:
        # 返回所有百分位数的平均Peak NRMSE
        return np.mean(list(peak_nrms_values.values()))

# 添加peak_nrms的大写别名
PEAK_NRMS = peak_nrms

def get_histogram(img):
    h, w = img.shape
    hist = [0.0] * 256
    for i in range(h):
        for j in range(w):
            hist[int(img[i, j])] += 1
    return np.array(hist) / float(h * w)

def normalize_exposure(img):
    img = img.astype(int)
    hist = get_histogram(img)
    cdf = np.array([sum(hist[:i+1]) for i in range(len(hist))])
    sk = np.uint8(255 * cdf)
    height, width = img.shape
    normalized = np.zeros_like(img)
    for i in range(height):
        for j in range(width):
            normalized[i, j] = sk[int(img[i, j])]
    return normalized.astype(int)

def tpr(tp, fn): return tp / (tp + fn)
def fpr(fp, tn): return fp / (fp + tn)
def precision(tp, fp): return tp / (tp + fp)

def calculate_all(csv_path):
    tpr_sum_List = []
    fpr_sum_List = []
    precision_sum_List = []
    threshold_remain_list = []
    num = 0
    tpr_sum = fpr_sum = precision_sum = 0

    with open(os.path.join(csv_path), 'r') as csv_file:
        first_flag = False
        for line in csv_file:
            threshold, idx, tn, fp, fn, tp = line.strip().split(',')
            if threshold not in threshold_remain_list:
                if first_flag and num != 0:
                    tpr_sum_List.append(tpr_sum / num)
                    fpr_sum_List.append(fpr_sum / num)
                    precision_sum_List.append(precision_sum / num)
                threshold_remain_list.append(threshold)
                tpr_sum = fpr_sum = precision_sum = 0
                num = 0
                first_flag = True

            if int(fp) == 0 and int(tn) == 0: continue
            elif int(tp) == 0 and int(fn) == 0: continue
            elif int(tp) == 0 and int(fp) == 0: continue
            else:
                tpr_sum += tpr(int(tp), int(fn))
                fpr_sum += fpr(int(fp), int(tn))
                precision_sum += precision(int(tp), int(fp))
                num += 1
        if num != 0:
            tpr_sum_List.append(tpr_sum / num)
            fpr_sum_List.append(fpr_sum / num)
            precision_sum_List.append(precision_sum / num)
    return tpr_sum_List, fpr_sum_List, precision_sum_List

def calculated_score(threshold_idx=None, temp_path=None, label_path=None, save_path=None, threshold_label=None, preds=None):
    file = open(os.path.join(temp_path, f'tpr_fpr_{threshold_idx}.csv'), 'w')
    f_csv = csv.writer(file, delimiter=',')
    for idx, pred in enumerate(preds):
        target_test = np.load(os.path.join(label_path, pred)).reshape(-1)
        target_probabilities = np.load(os.path.join(save_path, 'test_result', pred)).reshape(-1)
        target_test[target_test >= threshold_label] = 1
        target_test[target_test < threshold_label] = 0
        target_probabilities[target_probabilities >= threshold_idx] = 1
        target_probabilities[target_probabilities < threshold_idx] = 0

        if np.sum(target_probabilities == 0) == 0 and np.sum(target_test == 0) == 0:
            tp = 256 * 256
            tn = fn = fp = 0
        elif np.sum(target_probabilities == 1) == 0 and np.sum(target_test == 1) == 0:
            tn = 256 * 256
            tp = fn = fp = 0
        else:
            tn, fp, fn, tp = confusion_matrix(target_test, target_probabilities).ravel()
        f_csv.writerow([str(threshold_idx)] + [str(i) for i in [idx, tn, fp, fn, tp]])
    print(f'{threshold_idx}-done')

def multi_process_score(out_name=None, threshold=0.0, label_path=None, save_path=None):
    uid = str(uuid.uuid4())
    suid = ''.join(uid.split('-'))
    temp_path = f'./{suid}'
    psutil.cpu_percent(None)
    time.sleep(0.5)
    pool = mul.Pool(int(mul.cpu_count() * (1 - psutil.cpu_percent(None) / 100.0)))
    preds = [v for v in scandir(os.path.join(save_path, 'test_result'), suffix='npy', recursive=True)]
    os.makedirs(temp_path, exist_ok=True)
    threshold_list = np.linspace(0, 1, endpoint=False, num=200)
    calculated_score_parital = partial(calculated_score, temp_path=temp_path, label_path=label_path, save_path=save_path, threshold_label=threshold, preds=preds)
    pool.map(calculated_score_parital, threshold_list)
    print(f'{suid}')
    for list_i in threshold_list:
        with open(os.path.join(temp_path, f'tpr_fpr_{list_i}.csv'), 'r') as fr, open(os.path.join(temp_path, f'{out_name}'), 'a') as f:
            f.write(fr.read())
    print('copying')
    os.system(f'cp {os.path.join(temp_path, out_name)} {os.path.join(save_path, out_name)}')
    print('remove temp files')
    os.system(f'rm -rf {temp_path}')

def get_sorted_list(fpr_sum_List, tpr_sum_List):
    fpr_list = []
    tpr_list = []
    for i, j in zip(fpr_sum_List, tpr_sum_List):
        if i not in fpr_list:
            fpr_list.append(i)
            tpr_list.append(j)
    fpr_list.reverse()
    tpr_list.reverse()
    fpr_list, tpr_list = zip(*sorted(zip(fpr_list, tpr_list)))
    return fpr_list, tpr_list

def roc_prc(save_path):
    tpr_sum_List, fpr_sum_List, precision_sum_List = calculate_all(os.path.join(os.getcwd(), save_path, 'roc_prc.csv'))
    fpr_list, tpr_list = get_sorted_list(fpr_sum_List, tpr_sum_List)
    fpr_list = list(fpr_list) + [1]
    tpr_list = list(tpr_list) + [1]
    roc_numerator = 0
    for i in range(len(tpr_list) - 1):
        roc_numerator += (tpr_list[i] + tpr_list[i + 1]) * (fpr_list[i + 1] - fpr_list[i]) / 2
    tpr_list, p_list = get_sorted_list(tpr_sum_List, precision_sum_List)
    x_smooth = np.linspace(0, 1, 25)
    y_smooth = make_interp_spline(tpr_list, p_list, k=3)(x_smooth)
    prc_numerator = 0
    for i in range(len(y_smooth) - 1):
        prc_numerator += (y_smooth[i] + y_smooth[i + 1]) * (x_smooth[i + 1] - x_smooth[i]) / 2
    return roc_numerator, prc_numerator

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        if tensor.dim() == 4:  # [batch_size, C, H, W]
            tensor = [tensor[i] for i in range(tensor.size(0))]  # 拆成列表
        else:
            tensor = [tensor]  # 单张图片转为列表
    result = []
    for _tensor in tensor:
        _tensor = _tensor.float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = _tensor.dim()

        if n_dim == 3:  # [C, H, W]
            img_np = _tensor.numpy()
            img_np = np.transpose(img_np, (1, 2, 0))  # [H, W, C]
        elif n_dim == 2:  # [H, W]
            img_np = _tensor.numpy()[..., None]  # [H, W, 1]
        else:
            raise ValueError(f'Only support 3D or 2D tensor after batch split. '
                             f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    return result[0] if len(result) == 1 else result

def build_metric(metric_name):
    metric_name_lower = metric_name.lower()
    if metric_name_lower in ['peaknrms', 'peak_nrms', 'peak_nrmse']:
        return peak_nrms
    elif metric_name_lower in ['nrmse', 'nrms']:
        return nrms
    elif metric_name_lower == 'mae':
        return mae
    elif metric_name_lower == 'ssim':
        return ssim
    elif metric_name_lower == 'psnr':
        return psnr
    elif metric_name_lower == 'emd':
        return emd
    elif metric_name_lower in ['r2', 'r_squared']:
        return r2
    elif metric_name_lower in ['pearson', 'pear']:
        return pearson
    elif metric_name_lower in ['spearman', 'spea']:
        return spearman
    elif metric_name_lower in ['kendall', 'kend']:
        return kendall
    elif metric_name_lower == 'auc':
        return auc
    else:
        # 尝试从全局命名空间获取
        return globals().get(metric_name_lower, globals().get(metric_name))

def build_roc_prc_metric(threshold=None, dataroot=None, ann_file=None, save_path=None, **kwargs):
    if not ann_file:
        raise FileExistsError
    with open(ann_file, 'r') as fin:
        for line in fin:
            parts = line.strip().split(',')
            label = parts[-1] if len(parts) != 2 else parts[1]
            break
    label_name = label.split('/')[0]
    print(os.path.join(dataroot, label_name))
    multi_process_score(out_name='roc_prc.csv', threshold=threshold, label_path=os.path.join(dataroot, label_name), save_path=os.path.join('.', save_path))
    return roc_prc(save_path)


# 新增的评估指标

@input_converter(apply_to=('img1', 'img2'), out_type=np.float32)
def r2(img1, img2, crop_border=0):
    """
    计算R²决定系数 (R-squared)

    Args:
        img1 (np.ndarray): 真实值图像 (ground truth)
        img2 (np.ndarray): 预测值图像 (prediction)
        crop_border (int): 边界裁剪像素数

    Returns:
        float: R²值，范围通常在(-∞, 1]，1表示完美预测
    """
    assert img1.shape == img2.shape, f'Image shapes differ: {img1.shape}, {img2.shape}'
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    # img1是ground truth, img2是prediction
    y_true = img1.flatten()
    y_pred = img2.flatten()

    return r2_score(y_true, y_pred)


@input_converter(apply_to=('img1', 'img2'), out_type=np.float32)
def pearson(img1, img2, crop_border=0):
    """
    计算Pearson相关系数

    Args:
        img1 (np.ndarray): 真实值图像 (ground truth)
        img2 (np.ndarray): 预测值图像 (prediction)
        crop_border (int): 边界裁剪像素数

    Returns:
        float: Pearson相关系数，范围[-1, 1]
    """
    assert img1.shape == img2.shape, f'Image shapes differ: {img1.shape}, {img2.shape}'
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    y_true = img1.flatten()
    y_pred = img2.flatten()

    corr, _ = pearsonr(y_true, y_pred)
    return corr if not math.isnan(corr) else 0.0


@input_converter(apply_to=('img1', 'img2'), out_type=np.float32)
def spearman(img1, img2, crop_border=0):
    """
    计算Spearman相关系数

    Args:
        img1 (np.ndarray): 真实值图像 (ground truth)
        img2 (np.ndarray): 预测值图像 (prediction)
        crop_border (int): 边界裁剪像素数

    Returns:
        float: Spearman相关系数，范围[-1, 1]
    """
    assert img1.shape == img2.shape, f'Image shapes differ: {img1.shape}, {img2.shape}'
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    y_true = img1.flatten()
    y_pred = img2.flatten()

    corr, _ = spearmanr(y_true, y_pred)
    return corr if not math.isnan(corr) else 0.0


@input_converter(apply_to=('img1', 'img2'), out_type=np.float32)
def kendall(img1, img2, crop_border=0):
    """
    计算Kendall相关系数

    Args:
        img1 (np.ndarray): 真实值图像 (ground truth)
        img2 (np.ndarray): 预测值图像 (prediction)
        crop_border (int): 边界裁剪像素数

    Returns:
        float: Kendall相关系数，范围[-1, 1]
    """
    assert img1.shape == img2.shape, f'Image shapes differ: {img1.shape}, {img2.shape}'
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    y_true = img1.flatten()
    y_pred = img2.flatten()

    corr, _ = kendalltau(y_true, y_pred)
    return corr if not math.isnan(corr) else 0.0


@input_converter(apply_to=('img1', 'img2'), out_type=np.float32)
def auc(img1, img2, crop_border=0, threshold=0.5):
    """
    计算AUC (Area Under Curve)
    将回归问题转换为二分类问题来计算AUC

    Args:
        img1 (np.ndarray): 真实值图像 (ground truth)
        img2 (np.ndarray): 预测值图像 (prediction)
        crop_border (int): 边界裁剪像素数
        threshold (float): 二值化阈值，基于真实值的分位数

    Returns:
        float: AUC值，范围[0, 1]
    """
    assert img1.shape == img2.shape, f'Image shapes differ: {img1.shape}, {img2.shape}'
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    y_true = img1.flatten()
    y_pred = img2.flatten()

    # 使用真实值的中位数作为阈值进行二值化
    threshold_value = np.percentile(y_true, threshold * 100)
    y_true_binary = (y_true > threshold_value).astype(int)

    # 确保有两个类别
    if len(np.unique(y_true_binary)) < 2:
        return 0.5  # 如果只有一个类别，返回随机分类的AUC

    try:
        auc_score = roc_auc_score(y_true_binary, y_pred)
        return auc_score
    except ValueError:
        return 0.5


# 大写版本的指标别名，保持向后兼容
NRMS = nrms
SSIM = ssim
PSNR = psnr
MAE = mae
EMD = emd
R2 = r2
PEARSON = pearson
SPEARMAN = spearman
KENDALL = kendall
AUC = auc