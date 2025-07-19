# Copyright 2022 CircuitNet. All rights reserved.

from .gpdl import GPDL
from .routenet import RouteNet
from .mavi import MAVI
from .gat_congestion import GATCongestion
# 注释掉不存在的模块导入
# from .hybrid_model import HybridModel
from .gcn_congestion import GCNCongestion
from .graphsage_congestion import GraphSAGECongestion
from .rgcn_congestion import RGCNCongestion
from .cgan_congestion import CGANCongestion
from .fcn_congestion import FCNCongestion
from .lhnn_congestion import LHNNCongestion
# from .enhancedgpdl import GPDL_Enhanced
from .cvae import CVAECongestion
# 注释掉VisionMamba模型导入，避免mamba_ssm库的依赖问题
# from .vision_mamba_unet import VisionMambaUNet

from .swin_transformer import SwinTransformerCongestion
from .ibunet import Congestion_Prediction_Net, DRC_Prediction_Net

# 导入BRAU-Net++模型
try:
    from .braunetplus_congestion import BRAUNet_Congestion
    HAS_BRAUNETPLUS = True
except ImportError as e:
    HAS_BRAUNETPLUS = False
    # 不使用简化版本，按照用户要求
    # from .braunet_simplified import BRAUNet_Simplified

# 确保所有模型都在全局命名空间中可用
__all__ = [
    "GPDL",
    "RouteNet",
    "MAVI",
    "GATCongestion",
    "GCNCongestion",
    "GraphSAGECongestion",
    "RGCNCongestion",
    "CGANCongestion",
    "FCNCongestion",
    "LHNNCongestion",
    "CVAECongestion",
    "SwinTransformerCongestion",
    "Congestion_Prediction_Net",
    "DRC_Prediction_Net"
]

# 如果成功导入了BRAUNet_Congestion，则将其添加到__all__列表中
if HAS_BRAUNETPLUS:
    __all__.append("BRAUNet_Congestion")

# 将所有模型添加到模块的__dict__中，使其可以通过models.__dict__访问
import sys
current_module = sys.modules[__name__]
for model_name in __all__:
    if hasattr(current_module, model_name):
        model_class = getattr(current_module, model_name)
        current_module.__dict__[model_name] = model_class

# 导入EGE-UNet模型
from .egeunet_congestion import EGEUNetCongestion
