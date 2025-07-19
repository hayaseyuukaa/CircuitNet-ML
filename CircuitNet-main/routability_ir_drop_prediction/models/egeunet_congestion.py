#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EGE-UNet (Edge Guidance Enhanced U-Net) 模型实现
用于芯片布局布线拥塞预测任务

参考论文: "EGE-UNet: Edge Guidance Enhanced U-Net for Congestion Prediction in VLSI Placement"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """双卷积块：Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)


class EdgeAttentionBlock(nn.Module):
    """边缘注意力模块"""
    
    def __init__(self, channels):
        super(EdgeAttentionBlock, self).__init__()
        
        # 边缘检测卷积核
        self.edge_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        
        # 注意力生成模块
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 提取边缘特征
        edge_features = self.edge_conv(x)
        
        # 生成注意力权重
        attention_weights = self.attention(edge_features)
        
        # 应用注意力
        return x * attention_weights


class DownBlock(nn.Module):
    """下采样块：MaxPool -> ConvBlock"""
    
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)
        
    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    """上采样块：ConvTranspose -> Concat -> ConvBlock"""
    
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 处理输入特征图尺寸不匹配的情况
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # 拼接特征
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class EGEUNetCongestion(nn.Module):
    """EGE-UNet模型用于拥塞预测"""
    
    def __init__(self, in_channels=3, out_channels=1, filters_base=64):
        """
        初始化EGE-UNet模型
        
        参数:
            in_channels: 输入通道数，默认为3（单元密度、引脚密度、网线密度）
            out_channels: 输出通道数，默认为1（拥塞图）
            filters_base: 基础滤波器数量，默认为64
        """
        super(EGEUNetCongestion, self).__init__()
        
        # 编码器部分
        self.inc = ConvBlock(in_channels, filters_base)
        self.down1 = DownBlock(filters_base, filters_base * 2)
        self.down2 = DownBlock(filters_base * 2, filters_base * 4)
        self.down3 = DownBlock(filters_base * 4, filters_base * 8)
        
        # 瓶颈层
        self.bottleneck = DownBlock(filters_base * 8, filters_base * 16)
        
        # 边缘注意力模块
        self.edge_attention1 = EdgeAttentionBlock(filters_base * 8)
        self.edge_attention2 = EdgeAttentionBlock(filters_base * 4)
        self.edge_attention3 = EdgeAttentionBlock(filters_base * 2)
        self.edge_attention4 = EdgeAttentionBlock(filters_base)
        
        # 解码器部分
        self.up1 = UpBlock(filters_base * 16, filters_base * 8)
        self.up2 = UpBlock(filters_base * 8, filters_base * 4)
        self.up3 = UpBlock(filters_base * 4, filters_base * 2)
        self.up4 = UpBlock(filters_base * 2, filters_base)
        
        # 输出层
        self.outc = nn.Conv2d(filters_base, out_channels, kernel_size=1)
        
    def forward(self, x):
        # 编码器路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        
        # 应用边缘注意力
        x4_att = self.edge_attention1(x4)
        x3_att = self.edge_attention2(x3)
        x2_att = self.edge_attention3(x2)
        x1_att = self.edge_attention4(x1)
        
        # 解码器路径
        x = self.up1(x5, x4_att)
        x = self.up2(x, x3_att)
        x = self.up3(x, x2_att)
        x = self.up4(x, x1_att)
        
        # 输出层
        logits = self.outc(x)
        
        # 使用Sigmoid激活函数确保输出在[0,1]范围内
        return torch.sigmoid(logits)
