#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CircuitNet 测试脚本
基于ibUNet项目的inference.py改写
"""

from __future__ import print_function
import os
# 在导入PyTorch之前设置CUDA_VISIBLE_DEVICES，确保只使用GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 抑制常见警告
import warnings
warnings.filterwarnings("ignore", message=".*MMCV will release v2.0.0.*")
warnings.filterwarnings("ignore", message=".*A NumPy version.*is required for this version of SciPy.*")
warnings.filterwarnings("ignore", message=".*Importing from timm.models.layers is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os.path as osp
import json
import numpy as np
import sys
import torch
import gc
from tqdm import tqdm
from datasets.build_dataset import build_dataset
from utils.metrics import build_metric, build_roc_prc_metric
from models.build_model import build_model
from utils.configs import Parser





def test():
    """测试函数"""
    argp = Parser()
    arg = argp.parser.parse_args()
    
    arg_dict = vars(arg)
    if arg.arg_file is not None:
        with open(arg.arg_file, 'rt') as f:
            arg_dict.update(json.load(f))
    
    arg_dict['ann_file'] = arg_dict['ann_file_test'] 
    arg_dict['test_mode'] = True
    
    print('===> Building model')
    # Initialize model parameters
    model = build_model(arg_dict)

    # 设备选择 - 确保只使用单个GPU
    if not arg_dict.get('cpu', False):
        print("\n===> 使用GPU测试")

        # 确保CUDA可用并选择设备
        if torch.cuda.is_available():
            device = torch.device('cuda')  # 由于CUDA_VISIBLE_DEVICES='0'，这里会自动使用GPU 0
            print(f"✅ 使用GPU: {device}")
            print(f"✅ 可见GPU数量: {torch.cuda.device_count()}")
            print(f"✅ 当前CUDA设备: cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device('cpu')
            print("⚠️  CUDA不可用，回退到CPU")

        model = model.to(device)
        print(f"✅ 模型移动到 {device}")
    else:
        print("\n===> 使用CPU测试")
        device = torch.device('cpu')
        model = model.to(device)

    print('\n===> Loading datasets')
    # Initialize dataset
    dataset = build_dataset(arg_dict)
    
    # 加载预训练模型
    if arg_dict.get('pretrained'):
        print(f"===> 加载预训练模型: {arg_dict['pretrained']}")
        if device.type == 'cuda':
            # 强制映射到当前可见的GPU设备，避免设备不匹配错误
            checkpoint = torch.load(arg_dict['pretrained'], map_location=device)
        else:
            checkpoint = torch.load(arg_dict['pretrained'], map_location='cpu')

        # 处理不同的检查点格式
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("✅ 检测到包含state_dict的检查点格式")
            else:
                state_dict = checkpoint
                print("✅ 检测到直接state_dict格式")
        else:
            state_dict = checkpoint
            print("✅ 检测到其他格式检查点")

        # 尝试加载state_dict
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✅ 严格模式加载成功")
        except RuntimeError as e:
            print(f"❌ 严格模式加载失败: {str(e)[:200]}...")
            print("🔄 尝试非严格模式加载...")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"⚠️  缺失的键 ({len(missing_keys)}个): {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"⚠️  意外的键 ({len(unexpected_keys)}个): {unexpected_keys[:5]}...")
            print("✅ 非严格模式加载完成")
    
    model.eval()
    
    # Build metrics
    eval_metrics = arg_dict.get('eval_metric', ['NRMS', 'SSIM','PSNR'])
    metrics = {}
    for k in eval_metrics:
        metric_func = build_metric(k)
        if metric_func is not None:
            metrics[k] = metric_func
        else:
            print(f"⚠️  警告: 无法找到指标函数 '{k}'")

    avg_metrics = {k: 0 for k in metrics.keys()}
    
    count = 0
    print(f"\n===> 开始测试 ({len(dataset)} 个样本)")
    
    with torch.no_grad():
        with tqdm(total=len(dataset)) as bar:
            for feature, label, label_path in dataset:
                if device.type == 'cuda':
                    input, target = feature.to(device), label.to(device)
                else:
                    input, target = feature, label
                
                prediction = model(input)
                
                # 计算指标
                for metric, metric_func in metrics.items():
                    try:
                        if metric.lower() in ['peak_nrms', 'peaknrms']:
                            # 对于peak_nrms，直接传递tensor，让装饰器处理转换
                            detailed_result = metric_func(target.cpu(), prediction.squeeze(1).cpu(), return_dict=True)
                            avg_value = metric_func(target.cpu(), prediction.squeeze(1).cpu(), return_dict=False)

                            # 存储详细结果
                            if 'peak_nrms_detailed' not in avg_metrics:
                                avg_metrics['peak_nrms_detailed'] = {p: 0 for p in [0.5, 1, 2, 5]}

                            for p, value in detailed_result.items():
                                avg_metrics['peak_nrms_detailed'][p] += value

                            avg_metrics[metric] += avg_value
                        else:
                            metric_value = metric_func(target.cpu(), prediction.squeeze(1).cpu())
                            if metric_value != 1:  # 避免异常值
                                avg_metrics[metric] += metric_value
                    except Exception as e:
                        print(f"⚠️  指标 {metric} 计算失败: {e}")
                
                # 保存测试结果（如果需要）
                if arg_dict.get('plot_roc', False):
                    save_path = osp.join(arg_dict['save_path'], 'test_result')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    file_name = osp.splitext(osp.basename(label_path[0]))[0]
                    save_file = osp.join(save_path, f'{file_name}.npy')
                    output_final = prediction.float().detach().cpu().numpy()
                    np.save(save_file, output_final)
                    count += 1
                
                bar.update(1)
    
    # 输出平均指标
    print("\n===> 测试结果:")

    if len(dataset) == 0:
        print("  ⚠️  数据集为空，无法计算指标")
        return

    # 基础指标
    basic_metrics = ['NRMS', 'SSIM', 'PSNR', 'MAE']
    for metric in basic_metrics:
        if metric in avg_metrics:
            avg_value = avg_metrics[metric] / len(dataset)
            print(f"  {metric}: {avg_value:.4f}")

    # Peak NRMS详细结果（类似您的表格格式）
    if 'peak_nrms_detailed' in avg_metrics:
        print(f"\n  Peak NRMSE详细结果:")
        percentiles = [0.5, 1, 2, 5]
        header = "    " + "".join([f"{p:>8}%" for p in percentiles]) + f"{'average':>10}"
        print(header)

        values = []
        for p in percentiles:
            avg_val = avg_metrics['peak_nrms_detailed'][p] / len(dataset)
            values.append(avg_val)

        avg_all = sum(values) / len(values)
        values_str = "    " + "".join([f"{v:8.3f}" for v in values]) + f"{avg_all:10.3f}"
        print(values_str)

    # 相关性指标
    correlation_metrics = ['R2', 'PEARSON', 'SPEARMAN', 'KENDALL']
    correlation_found = False
    for metric in correlation_metrics:
        if metric in avg_metrics:
            if not correlation_found:
                print(f"\n  相关性指标:")
                correlation_found = True
            avg_value = avg_metrics[metric] / len(dataset)
            print(f"    {metric}: {avg_value:.4f}")

    # 其他指标
    other_metrics = ['AUC']
    for metric in other_metrics:
        if metric in avg_metrics:
            avg_value = avg_metrics[metric] / len(dataset)
            print(f"  {metric}: {avg_value:.4f}")

    # 显示所有其他未分类的指标
    displayed_metrics = set(basic_metrics + ['peak_nrms_detailed'] + correlation_metrics + other_metrics)
    for metric, avg_metric in avg_metrics.items():
        if metric not in displayed_metrics:
            avg_value = avg_metric / len(dataset)
            print(f"  {metric}: {avg_value:.4f}")
    
    # ROC分析（如果需要）
    if arg_dict.get('plot_roc', False):
        try:
            roc_metric, _ = build_roc_prc_metric(**arg_dict)
            print(f"\n===> AUC of ROC: {roc_metric:.4f}")
        except Exception as e:
            print(f"⚠️  ROC分析失败: {e}")
    
    print(f"\n===> 测试完成，共处理 {len(dataset)} 个样本")
    if count > 0:
        print(f"===> 保存了 {count} 个预测结果")


if __name__ == "__main__":
    test()
