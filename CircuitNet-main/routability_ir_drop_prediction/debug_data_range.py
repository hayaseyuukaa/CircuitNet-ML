#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试数据范围的脚本
"""

import numpy as np
import torch
import sys
import os
import json
sys.path.append(os.path.dirname(__file__))

from datasets.build_dataset import build_dataset
from models.build_model import build_model
from utils.configs import Parser

def debug_data_range():
    """调试数据范围"""
    print("=== 调试数据范围 ===")
    
    # 模拟RouteNet N28测试配置
    arg_dict = {
        "test_mode": True,
        "model_type": "RouteNet",
        "in_channels": 3,
        "out_channels": 1,
        "dataroot": "/doc/gky/CircuitNet6/CircuitNet-main/routability_ir_drop_prediction/training_set/congestion_trainingset1.0/congestion_trainingset/congestion",
        "ann_file": "/doc/gky/CircuitNet6/CircuitNet-main/routability_ir_drop_prediction/files/test_N28.csv",
        "dataset_type": "CongestionDataset"
    }
    
    print('===> Loading datasets')
    dataset = build_dataset(arg_dict)
    
    print(f"数据集大小: {len(dataset)}")
    
    # 检查前几个样本的数据范围
    print("\n===> 检查数据范围 (前5个样本)")
    
    for i in range(min(5, len(dataset))):
        feature, label, label_path = dataset[i]
        
        print(f"\n样本 {i+1}:")
        print(f"  特征形状: {feature.shape}")
        print(f"  标签形状: {label.shape}")
        print(f"  特征范围: [{feature.min():.4f}, {feature.max():.4f}]")
        print(f"  标签范围: [{label.min():.4f}, {label.max():.4f}]")
        print(f"  特征均值: {feature.mean():.4f}")
        print(f"  标签均值: {label.mean():.4f}")
        print(f"  标签路径: {label_path}")
        
        # 检查标签的分布
        label_np = label.numpy()
        print(f"  标签统计:")
        print(f"    25%分位数: {np.percentile(label_np, 25):.4f}")
        print(f"    50%分位数: {np.percentile(label_np, 50):.4f}")
        print(f"    75%分位数: {np.percentile(label_np, 75):.4f}")
        print(f"    95%分位数: {np.percentile(label_np, 95):.4f}")
        print(f"    99%分位数: {np.percentile(label_np, 99):.4f}")

if __name__ == "__main__":
    debug_data_range()
