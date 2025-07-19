import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import pathlib
import platform

# 添加BRAU-Net++路径到系统路径
BRAUNETPLUS_PATH = "/doc/gky/CircuitNet6/BRAU-Netplusplus"
if BRAUNETPLUS_PATH not in sys.path:
    sys.path.append(BRAUNETPLUS_PATH)

# 根据平台类型修复pathlib问题
if platform.system() == 'Windows':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

# 导入BRAU-Net++模型
from networks.bra_unet import BRAUnet

class BRAUNet_Congestion(nn.Module):
    """BRAU-Net++ 模型适配器，用于CircuitNet拥塞预测任务
    
    将BRAU-Net++模型集成到CircuitNet框架中，使其符合CircuitNet的接口约定。
    
    Args:
        in_channels (int): 输入通道数，默认为3（单元密度、线密度、引脚密度）
        out_channels (int): 输出通道数，默认为1（拥塞热图）
        img_size (int): 输入图像大小，默认为256
        n_win (int): Bi-level路由注意力窗口数，默认为8
    """
    def __init__(self, in_channels=3, out_channels=1, img_size=256, n_win=8, **kwargs):
        super(BRAUNet_Congestion, self).__init__()
        
        # 创建BRAU-Net++模型实例
        self.braunet = BRAUnet(
            img_size=img_size,
            in_chans=in_channels,
            num_classes=out_channels,
            n_win=n_win
        )
        
    def forward(self, x):
        # BRAU-Net++要求输入图像通道必须为3
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # 获取模型输出
        logits = self.braunet(x)
        
        # 如果需要，可以在这里添加额外的处理或输出转换
        return logits
    
    def init_weights(self, pretrained=None, strict=True):
        """初始化模型权重
        
        Args:
            pretrained (str, optional): 预训练权重文件路径，如果为None则随机初始化
            strict (bool, optional): 是否严格加载权重
        """
        if pretrained is not None:
            print(f"加载预训练权重: {pretrained}")
            if os.path.exists(pretrained):
                try:
                    state_dict = torch.load(pretrained, map_location='cpu')
                    # 如果权重存储在'state_dict'键中，提取它
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    
                    # 检查是否需要调整模型权重的键名
                    model_keys = set(self.state_dict().keys())
                    pretrained_keys = set(state_dict.keys())
                    
                    # 如果键名不匹配，尝试处理前缀差异
                    if not any(k in pretrained_keys for k in model_keys):
                        # 尝试去除模型权重键名前缀
                        new_state_dict = {}
                        for k, v in state_dict.items():
                            # 处理可能的前缀差异
                            if k.startswith('module.'):
                                name = k[7:]  # 去除 'module.' 前缀
                            elif k.startswith('braunet.bra_unet.'):
                                name = k[17:]  # 去除 'braunet.bra_unet.' 前缀
                            elif k.startswith('bra_unet.'):
                                name = k[9:]  # 去除 'bra_unet.' 前缀
                            else:
                                name = k
                            new_state_dict[name] = v
                        state_dict = new_state_dict
                    
                    # 加载权重
                    msg = self.braunet.load_state_dict(state_dict, strict=strict)
                    print(f"预训练权重加载结果: {msg}")
                except Exception as e:
                    print(f"加载预训练权重失败: {e}")
            else:
                print(f"预训练权重文件不存在: {pretrained}")
                print("使用默认随机初始化权重")
        else:
            # 自定义初始化方法，不依赖原始BRAU-Net++的load_from
            print("使用随机初始化权重")
            # 执行标准的权重初始化
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0) 