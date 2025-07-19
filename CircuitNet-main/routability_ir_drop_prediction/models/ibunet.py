# Copyright 2022 CircuitNet. All rights reserved.
# ibUNet: Inception Boosted U-Net Neural Network for Routability Prediction

import torch
import torch.nn as nn
from collections import OrderedDict

# 全局变量用于控制任务类型
g_prediction_task = "UnKnown"  # "Congestion" or "DRC"


def generation_init_weights(module):
    """权重初始化函数 - 与原始ibUNet完全一致"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                    or classname.find('Linear') != -1):

            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    module.apply(init_func)


class create_encoder_single_conv(nn.Module):
    """Inception编码器单卷积层 - 与原始ibUNet完全一致"""
    def __init__(self, in_chs, out_chs, kernel):
        super().__init__()
        global g_prediction_task
        assert kernel % 2 == 1
        if ("DRC" == g_prediction_task):
            self.single_Conv = nn.Sequential(
                nn.Conv2d(in_chs, out_chs, kernel_size=kernel, padding=(kernel - 1) // 2),
                nn.InstanceNorm2d(out_chs, affine=True),
                nn.PReLU(num_parameters=out_chs)
            )
        elif ("Congestion" == g_prediction_task):
            self.single_Conv = nn.Sequential(
                nn.Conv2d(in_chs, out_chs, kernel_size=kernel, padding=(kernel - 1) // 2),
                nn.BatchNorm2d(out_chs),
                nn.PReLU(num_parameters=out_chs)
            )
        else:
            print("ERROR on prediction type!")

    def forward(self, x):
        out = self.single_Conv(x)
        return out


class EncoderInceptionModuleSignle(nn.Module):
    """Inception模块单元 - 与原始ibUNet完全一致"""
    def __init__(self, channels):
        assert channels % 2 == 0
        super().__init__()
        # put bottle-neck layers before convolution
        bn_ch = channels // 2
        self.bottleneck = create_encoder_single_conv(channels, bn_ch, 1)
        # bn -> Conv1, 3, 5, 7
        self.conv1 = create_encoder_single_conv(bn_ch, channels, 1)
        self.conv3 = create_encoder_single_conv(bn_ch, channels, 3)
        self.conv5 = create_encoder_single_conv(bn_ch, channels, 5)
        self.conv7 = create_encoder_single_conv(bn_ch, channels, 7)

        # pool-proj(no-bottle neck)
        self.pool3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(5, stride=1, padding=2)

    def forward(self, x):
        # Original inception is concatenation, but use simple addition instead
        bn = self.bottleneck(x)
        out = self.conv1(bn) + self.conv3(bn) + self.conv5(bn) + self.conv7(bn) + self.pool3(x) + self.pool5(x)
        return out


class EncoderModule(nn.Module):
    """编码器模块 - 与原始ibUNet完全一致"""
    def __init__(self, chs, repeat_num, use_inception):
        super().__init__()
        if use_inception:
            layers = [EncoderInceptionModuleSignle(chs) for i in range(repeat_num)]
        else:
            layers = [create_encoder_single_conv(chs, chs, 3) for i in range(repeat_num)]
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


class IncepEncoder(nn.Module):
    """Inception编码器 - 与原始ibUNet完全一致"""
    def __init__(self, use_inception, repeat_per_module, prediction_type, middel_layer_size=256):
        global g_prediction_task
        super().__init__()
        g_prediction_task = prediction_type
        self.encoderPart = EncoderModule(middel_layer_size, repeat_per_module, use_inception)

    def forward(self, x):
        out = self.encoderPart(x)
        return out

    def init_weights(self):
        """Initialize the weights."""
        generation_init_weights(self)


def load_state_dict(module, state_dict, strict=False, logger=None):
    """自定义状态字典加载函数"""
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None

    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    return missing_keys


class DoubleConv(nn.Module):
    """(convolution => [BN] => PReLU) * 2 - 与原始ibUNet完全一致"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        global g_prediction_task
        if not mid_channels:
            mid_channels = out_channels

        if ("DRC" == g_prediction_task):
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(mid_channels, affine=True),
                nn.PReLU(num_parameters=mid_channels),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.PReLU(num_parameters=out_channels)
            )
        elif ("Congestion" == g_prediction_task):
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.PReLU(num_parameters=mid_channels),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.PReLU(num_parameters=out_channels)
            )
        else:
            print("ERROR on prediction task!!")

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv - 与原始ibUNet完全一致"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            pass
            # 原始ibUNet注释：no advantage on our case ! Harley 2023/11/25
            # self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """ibUNet架构实现 - 与原始ibUNet完全一致"""
    def __init__(self, n_channels, n_classes, bilinear=True, task="UnKnown"):
        global g_prediction_task
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        # 动态缩放因子：ISPD数据集(6通道)使用标准架构(_SCALE_=1)，N14/N28数据集(3通道)使用缩放架构(_SCALE_=2)
        _SCALE_ = 1 if n_channels == 6 else 2
        g_prediction_task = task

        self.inc = DoubleConv(n_channels, 64//_SCALE_)
        self.down1 = Down(64//_SCALE_, 128//_SCALE_)
        self.down2 = Down(128//_SCALE_, 256//_SCALE_)
        self.down3 = Down(256//_SCALE_, 512//_SCALE_)
        self.down4 = Down(512//_SCALE_, 512//_SCALE_)
        self.up1 = Up(1024//_SCALE_, 256//_SCALE_, bilinear)
        self.up2 = Up(512//_SCALE_, 128//_SCALE_, bilinear)
        self.up3 = Up(256//_SCALE_, 64//_SCALE_, bilinear)
        self.up4 = Up(128//_SCALE_, 64//_SCALE_, bilinear)
        self.outc = OutConv(64//_SCALE_, n_classes)

        # 关键：添加Inception模块增强，与原始ibUNet一致
        self.incepEncoder = IncepEncoder(True, 1, g_prediction_task, 512//_SCALE_)
        self.Conv4Inception = DoubleConv(512//_SCALE_, 512//_SCALE_)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5B = self.incepEncoder(x5)  # Insert the Inception Module at the bottleneck
        x5C = self.Conv4Inception(x5B)
        x = self.up1(x5C, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def init_weights(self):
        """Initialize the weights."""
        generation_init_weights(self)


class PredictionNet(nn.Module):
    """预测网络基类"""
    def __init__(self, in_channels=3, out_channels=1, **kwargs):
        super().__init__()

    def forward(self, x):
        pass
        
    def init_weights(self, pretrained=None, strict=True, **kwargs):
        """权重初始化方法"""
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
        elif pretrained is None:
            generation_init_weights(self)
        else:
            raise TypeError("'pretrained' must be a str or None. ")


class Congestion_Prediction_Net(PredictionNet):
    """拥塞预测网络 - ibUNet实现，支持动态通道数配置"""
    def __init__(self, in_channels=3, out_channels=1, **kwargs):
        super().__init__()
        # 支持动态通道数：N14/N28使用3通道，ISPD使用6通道
        self.uNet = UNet(n_channels=in_channels, n_classes=out_channels, bilinear=True, task="Congestion")

    def forward(self, x):
        return self.uNet(x)

    def init_weights(self, pretrained=None, strict=True, **kwargs):
        super().init_weights(pretrained, strict, **kwargs)


class DRC_Prediction_Net(PredictionNet):
    """DRC预测网络 - ibUNet实现，与原始ibUNet完全一致"""
    def __init__(self, in_channels=9, out_channels=1, **kwargs):
        super().__init__()
        # 与原始ibUNet一致：使用固定的通道数9
        self.uNet = UNet(n_channels=9, n_classes=1, bilinear=True, task="DRC")

    def forward(self, x):
        return self.uNet(x)

    def init_weights(self, pretrained=None, strict=True, **kwargs):
        super().init_weights(pretrained, strict, **kwargs)