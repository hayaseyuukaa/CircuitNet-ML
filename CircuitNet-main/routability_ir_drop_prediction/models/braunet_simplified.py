import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 确保reduction不会导致通道数为0
        reduction_ratio = max(1, in_channels // reduction)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class CBAM(nn.Module):
    """CBAM注意力机制
    
    论文：'CBAM: Convolutional Block Attention Module'
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ConvBlock(nn.Module):
    """双卷积块"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        print(f"创建卷积块: 输入通道 = {in_channels}, 输出通道 = {out_channels}")
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    """下采样块"""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    """上采样块"""
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        # 上采样
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 注意力模块 - 应用于skip connection特征
        self.attention = CBAM(out_channels)
        
        # 卷积层 - 输入通道数是上采样后的特征通道加上跳跃连接的特征通道
        # 上采样后的特征通道与输入通道相同
        self.conv = ConvBlock(in_channels + out_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 输入尺寸可能不匹配，需要进行裁剪
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 应用注意力
        x2 = self.attention(x2)
        
        # 打印拼接前的特征形状
        print(f"拼接特征形状: x1 = {x1.shape}, x2 = {x2.shape}")
        
        # 拼接并卷积
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """输出卷积层"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class BRAUNet_Simplified(nn.Module):
    """简化版本的 BRAU-Net++，用于拥塞预测
    
    这是一个基于U-Net架构的简化实现，融合了一些BRAU-Net++的思想，
    但不依赖于原始BRAU-Net++的复杂组件和依赖项。
    
    Args:
        in_channels (int): 输入通道数，默认为3（单元密度、线密度、引脚密度）
        out_channels (int): 输出通道数，默认为1（拥塞热图）
        base_filters (int): 基础特征通道数，默认为64
    """
    def __init__(self, in_channels=3, out_channels=1, base_filters=32, **kwargs):
        super(BRAUNet_Simplified, self).__init__()
        # 减少基础通道数，以避免内存不足问题
        self.f = base_filters  # 默认值修改为32
        
        # 编码器路径
        self.inc = ConvBlock(in_channels, self.f)
        self.down1 = DownBlock(self.f, self.f*2)
        self.down2 = DownBlock(self.f*2, self.f*4)
        self.down3 = DownBlock(self.f*4, self.f*8)
        self.down4 = DownBlock(self.f*8, self.f*16)
        
        # 增强瓶颈层
        self.bottle_attention = CBAM(self.f*16)
        
        # 解码器路径
        self.up1 = UpBlock(self.f*16, self.f*8)
        self.up2 = UpBlock(self.f*8, self.f*4)
        self.up3 = UpBlock(self.f*4, self.f*2)
        self.up4 = UpBlock(self.f*2, self.f)
        self.outc = OutConv(self.f, out_channels)
        
        # 权重初始化
        self.apply(self._init_weights)
        
        # 打印模型架构通道数
        print(f"模型架构 - 基础通道数: {self.f}")
        print(f"编码器通道数: {self.f} -> {self.f*2} -> {self.f*4} -> {self.f*8} -> {self.f*16}")
        print(f"解码器通道数: {self.f*16} -> {self.f*8} -> {self.f*4} -> {self.f*2} -> {self.f} -> {out_channels}")
    
    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 处理单通道输入
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        # 打印输入张量形状
        print(f"输入形状: {x.shape}")
        
        # 编码器路径
        x1 = self.inc(x)           # 输出: f
        print(f"编码器特征 1: {x1.shape}")
        
        x2 = self.down1(x1)        # 输出: f*2
        print(f"编码器特征 2: {x2.shape}")
        
        x3 = self.down2(x2)        # 输出: f*4
        print(f"编码器特征 3: {x3.shape}")
        
        x4 = self.down3(x3)        # 输出: f*8
        print(f"编码器特征 4: {x4.shape}")
        
        x5 = self.down4(x4)        # 输出: f*16
        print(f"编码器特征 5 (瓶颈层): {x5.shape}")
        
        # 瓶颈层增强
        x5 = self.bottle_attention(x5)
        
        # 解码器路径
        x = self.up1(x5, x4)       # 输入: f*16, f*8  -> 输出: f*8
        print(f"解码器特征 1: {x.shape}")
        
        x = self.up2(x, x3)        # 输入: f*8, f*4   -> 输出: f*4
        print(f"解码器特征 2: {x.shape}")
        
        x = self.up3(x, x2)        # 输入: f*4, f*2   -> 输出: f*2
        print(f"解码器特征 3: {x.shape}")
        
        x = self.up4(x, x1)        # 输入: f*2, f     -> 输出: f
        print(f"解码器特征 4: {x.shape}")
        
        logits = self.outc(x)      # 输入: f          -> 输出: out_channels
        print(f"输出形状: {logits.shape}")
        
        return logits
    
    def init_weights(self, pretrained=None, strict=True, **kwargs):
        """符合CircuitNet框架的权重初始化接口"""
        if pretrained is not None:
            print(f"加载预训练权重: {pretrained}")
            try:
                state_dict = torch.load(pretrained, map_location='cpu')
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                self.load_state_dict(state_dict, strict=strict)
                print("预训练权重加载成功")
            except Exception as e:
                print(f"加载预训练权重失败: {e}")
                print("使用默认初始化权重")
        else:
            print("使用默认Kaiming初始化权重") 