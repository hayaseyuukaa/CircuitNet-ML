#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新增评估指标的脚本
"""

import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))

from utils.metrics import build_metric

def test_new_metrics():
    """测试新增的评估指标"""
    print("=== 测试新增评估指标 ===")
    
    # 创建测试数据
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 模拟真实的拥塞预测数据范围
    # 假设数据范围在[0, 1]之间（归一化后）
    target = torch.rand(1, 64, 64, dtype=torch.float32)
    
    # 创建有相关性的预测数据
    noise = torch.normal(0, 0.1, target.shape)
    prediction = target * 0.8 + noise * 0.2  # 80%相关性 + 20%噪声
    
    print(f"目标数据范围: [{target.min():.4f}, {target.max():.4f}]")
    print(f"预测数据范围: [{prediction.min():.4f}, {prediction.max():.4f}]")
    print(f"目标数据均值: {target.mean():.4f}")
    print(f"预测数据均值: {prediction.mean():.4f}")
    
    # 测试所有指标
    metrics_to_test = [
        'NRMS', 'SSIM', 'PSNR', 'MAE', 'peak_nrms',
        'r2', 'pearson', 'spearman', 'kendall', 'auc'
    ]
    
    print(f"\n=== 评估指标结果 ===")
    results = {}
    
    for metric_name in metrics_to_test:
        try:
            metric_func = build_metric(metric_name)
            if metric_func is None:
                print(f"  {metric_name}: 未找到指标函数")
                continue
                
            result = metric_func(target, prediction)
            results[metric_name] = result
            print(f"  {metric_name}: {result:.6f}")
            
        except Exception as e:
            print(f"  {metric_name}: 错误 - {e}")
    
    # 测试不同数据范围的影响
    print(f"\n=== 测试不同数据范围的影响 ===")
    
    # 测试大数值范围的数据（模拟未归一化的数据）
    target_large = target * 1000  # 放大1000倍
    prediction_large = prediction * 1000
    
    print(f"大数值范围数据:")
    print(f"  目标数据范围: [{target_large.min():.1f}, {target_large.max():.1f}]")
    print(f"  预测数据范围: [{prediction_large.min():.1f}, {prediction_large.max():.1f}]")
    
    for metric_name in ['MAE', 'NRMS', 'r2', 'pearson']:
        try:
            metric_func = build_metric(metric_name)
            result = metric_func(target_large, prediction_large)
            print(f"  {metric_name} (大数值): {result:.6f}")
        except Exception as e:
            print(f"  {metric_name} (大数值): 错误 - {e}")
    
    # 显示表格格式的结果
    print(f"\n=== 表格格式结果 ===")
    print("Setting                    | NRMSE↓ | SSIM↑ | R²↑   | PSNR↑ | Pear↑ | Spea↑ | Kend↑ | AUC↑  | MAE↓")
    print("-" * 95)
    
    nrmse = results.get('NRMS', 0)
    ssim = results.get('SSIM', 0)
    r2 = results.get('r2', 0)
    psnr = results.get('PSNR', 0)
    pearson = results.get('pearson', 0)
    spearman = results.get('spearman', 0)
    kendall = results.get('kendall', 0)
    auc = results.get('auc', 0)
    mae = results.get('MAE', 0)
    
    print(f"Test Result                | {nrmse:.3f}  | {ssim:.3f} | {r2:.3f} | {psnr:.1f} | {pearson:.3f} | {spearman:.3f} | {kendall:.3f} | {auc:.3f} | {mae:.1f}")
    
    print(f"\n=== 测试完成 ===")

if __name__ == "__main__":
    test_new_metrics()
