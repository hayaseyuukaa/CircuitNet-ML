import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_samples(feature, label, pred, save_path):
    """
    绘制特征图、标签和预测结果的可视化图像
    
    Args:
        feature (numpy.ndarray): 输入特征，形状为 [C, H, W]
        label (numpy.ndarray): 真实标签，形状为 [H, W]
        pred (numpy.ndarray): 预测结果，形状为 [H, W]
        save_path (str): 保存路径
    """
    # 创建保存路径的目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 创建图形
    plt.figure(figsize=(15, 5))
    
    # 绘制特征图 (只显示第一个通道)
    plt.subplot(1, 3, 1)
    plt.title('输入特征')
    plt.imshow(feature[0], cmap='viridis')
    plt.colorbar()
    
    # 绘制真实标签
    plt.subplot(1, 3, 2)
    plt.title('真实拥塞')
    plt.imshow(label, cmap='hot')
    plt.colorbar()
    
    # 绘制预测结果
    plt.subplot(1, 3, 3)
    plt.title('预测拥塞')
    plt.imshow(pred, cmap='hot')
    plt.colorbar()
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() 