# Copyright 2022 CircuitNet. All rights reserved.

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

import json
import torch
import torch.optim as optim
from tqdm import tqdm
import gc

from datasets.build_dataset import build_dataset
from utils.losses import build_loss
from models.build_model import build_model
from utils.configs import Parser
from utils.lr_scheduler import CosineRestartLr
from math import cos, pi
from datetime import datetime








def checkpoint(model, epoch, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_out_path = f"{save_path}/model_iters_{epoch}.pth"
    torch.save({'state_dict': model.state_dict()}, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


# CosineRestartLr类现在从utils.lr_scheduler导入，与原始ibUNet完全一致


def train():
    now = datetime.now()

    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)
    if arg.arg_file is not None:
        with open(arg.arg_file, "rt") as f:
            arg_dict.update(json.load(f))

    # 确保某些参数是整数类型
    arg_dict["max_iters"] = int(arg_dict["max_iters"])

    if not os.path.exists(arg_dict["save_path"]):
        os.makedirs(arg_dict["save_path"])
    with open(os.path.join(arg_dict["save_path"], "arg.json"), "wt") as f:
        json.dump(arg_dict, f, indent=4)

    arg_dict["ann_file"] = arg_dict["ann_file_train"]
    arg_dict["test_mode"] = False

    # 模型名称映射
    model_display_names = {
        "Congestion_Prediction_Net": "ibUNet (Inception Boosted U-Net)",
        "GPDL": "GPDL",
        "GCNCongestion": "GCN",
        "EGEUNetCongestion": "EGE-UNet",
        "GraphSAGECongestion": "GraphSAGE",
        "RGCNCongestion": "RGCN",
        "RouteNet": "RouteNet",
        "SwinTransformerCongestion": "Swin Transformer",
        "VMUNet": "VM-UNet",
        "BRAUNet_Congestion": "BRAU-Net++",
        "GATCongestion": "GAT"
    }

    model_type = arg_dict.get('model_type', 'Unknown')
    display_name = model_display_names.get(model_type, model_type)

    # 显示训练信息
    print("=" * 50)
    print("CircuitNet 训练开始")
    print("=" * 50)
    print(f"模型: {display_name}")
    print(f"工作目录: {arg_dict['save_path']}")
    print(f"配置: batch_size={arg_dict['batch_size']}, lr={arg_dict['lr']}, max_iters={arg_dict['max_iters']}")
    print("=" * 50)

    print(f"===> Building model: {display_name}")
    # Initialize model parameters
    model = build_model(arg_dict)

    # 设备选择和模型部署 - 确保只使用单个GPU
    if not arg_dict["cpu"]:
        print("\n===> 使用GPU训练")

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
        print("\n===> 使用CPU训练")
        device = torch.device('cpu')
        model = model.to(device)

    # 更新arg_dict中的设备信息
    arg_dict["device"] = device

    # 在设备选择完成后构建数据集
    print("\n===> Loading datasets")

    # 禁用pin_memory避免多GPU上下文初始化
    # pin_memory可能导致在多个GPU上创建CUDA上下文
    arg_dict["use_pin_memory"] = False
    

    # Initialize dataset
    dataset = build_dataset(arg_dict)

    # 简化数据集信息显示
    try:
        import pandas as pd
        train_csv = pd.read_csv(arg_dict['ann_file_train'])
        print(f"数据集: {len(train_csv)} 个训练样本")
    except Exception as e:
        print(f"数据集信息获取失败: {e}")

    # Build loss
    loss = build_loss(arg_dict)

    # Build Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=arg_dict["lr"],
        betas=(0.9, 0.999),
        weight_decay=arg_dict["weight_decay"],
    )

    # Build lr scheduler
    cosine_lr = CosineRestartLr(arg_dict["lr"], [arg_dict["max_iters"]], [1], 1e-7)
    cosine_lr.set_init_lr(optimizer)

    epoch_loss = 0
    iter_num = 0
    print_freq = 100
    save_freq = 10000
    Show_freq = 1000

    lossList = []

    while iter_num < arg_dict['max_iters']:
        # 智能进度条：在日志文件中简洁，在终端中美观
        import sys
        is_logging_to_file = not sys.stdout.isatty()

        if is_logging_to_file:
            # 日志文件模式：不显示进度条，只显示关键信息
            pbar = None
        else:
            # 终端模式：显示美观的进度条
            pbar = tqdm(total=print_freq, desc=f"Iter {iter_num//print_freq + 1}")

        batch_count = 0
        for feature, label, _ in dataset:
            # 使用设备
            device = arg_dict.get("device", torch.device('cpu'))

            try:
                input, target = feature.to(device), label.to(device)

                # 在第一次迭代时输出设备信息用于调试
                if iter_num == 1:
                    print(f"🔍 数据设备信息: input在{input.device}, target在{target.device}, 模型在{next(model.parameters()).device}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n❌ 数据移动时显存不足: {e}")
                    print("🔄 清理显存并重试...")

                    # 清理显存
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()

                    # 重试数据移动
                    try:
                        input, target = feature.to(device), label.to(device)
                    except RuntimeError:
                        print("⚠️  显存仍然不足，回退到CPU")
                        device = torch.device('cpu')
                        model = model.to(device)
                        arg_dict["device"] = device
                        input, target = feature.to(device), label.to(device)
                else:
                    raise e

            regular_lr = cosine_lr.get_regular_lr(iter_num)
            cosine_lr._set_lr(optimizer, regular_lr)

            prediction = model(input)

            optimizer.zero_grad()

            pixel_loss = loss(prediction, target)
            epoch_loss += pixel_loss.item()
            pixel_loss.backward()
            optimizer.step()

            iter_num += 1
            batch_count += 1

            # 更新进度显示
            if pbar:
                pbar.update(1)
            elif batch_count % 25 == 0:  # 日志模式：每25个batch显示一次进度
                progress = (batch_count / print_freq) * 100
                print(f"    训练进度: {batch_count}/{print_freq} ({progress:.1f}%)")

            if (iter_num % print_freq == 0 or iter_num==100):
                break

        # 关闭进度条
        if pbar:
            pbar.close()

        # 显示训练进度和显存使用情况
        avg_loss = epoch_loss / print_freq
        device = arg_dict.get("device", torch.device('cpu'))

        if device.type == 'cuda':
            try:
                # 确保查询正确的设备
                current_device = torch.cuda.current_device()
                target_device_id = device.index if device.index is not None else 0

                allocated_memory = torch.cuda.memory_allocated(device) / 1024**3
                reserved_memory = torch.cuda.memory_reserved(device) / 1024**3

                # 添加设备信息用于调试
                print("===> Iters({}/{}): Loss: {:.4f} | GPU显存: {:.1f}GB已分配, {:.1f}GB已保留 | 设备: {} (当前: cuda:{})".format(
                    iter_num, arg_dict['max_iters'], avg_loss, allocated_memory, reserved_memory, device, current_device))

                
            
            except Exception as e:
                print("===> Iters({}/{}): Loss: {:.4f} | GPU显存查询失败: {}".format(
                    iter_num, arg_dict['max_iters'], avg_loss, e))
        else:
            print("===> Iters({}/{}): Loss: {:.4f} | 使用CPU训练".format(
                iter_num, arg_dict['max_iters'], avg_loss))

        oneValue = avg_loss
        lossList.append(oneValue)
        if(len(lossList)>10):
            lossList.pop(0)
            sumValue =0
            for kk in range(len(lossList)):
                sumValue += lossList[kk]
            print("===> Average Loss: {:.4f}".format(sumValue / len(lossList)))

        if ((iter_num % save_freq == 0) or (100==iter_num)):
            checkpoint(model, iter_num, arg_dict['save_path'])

        if ((iter_num % Show_freq == 0) or (100==iter_num)):
            later = datetime.now()
            difference = (later - now).total_seconds()
            difference_minutes = int(difference // 60)
            difference_seconds = int(difference % 60)
            print('===> So far taining takes: '
                  + str(difference_minutes) + "(minutes) and "
                  + str(difference_seconds) + "(seconds)")

            max_iters = arg_dict['max_iters']
            expect_difference = difference*(max_iters/iter_num)
            difference_minutes = int(expect_difference // 60)
            difference_seconds = int(expect_difference % 60)
            print('===> Expected Time: '
                  + str(difference_minutes) + "(minutes) and "
                  + str(difference_seconds) + "(seconds)")

        epoch_loss = 0


if __name__ == "__main__":
    train()
