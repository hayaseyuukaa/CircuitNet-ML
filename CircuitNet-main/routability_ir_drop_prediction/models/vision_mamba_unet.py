import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from mamba_ssm import Mamba

def generation_init_weights(module):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                    or classname.find('Linear') != -1):
            
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    module.apply(init_func)

def load_state_dict(module, state_dict, strict=False, logger=None):
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

# 官方风格VSSBlock
class VSSBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=3, expand=2, drop_path=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # x: [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        x = self.norm(x)
        x = self.mamba(x)
        x = self.drop_path(x)
        x = self.mlp(x)
        # 恢复回[B, C, H, W]
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        return x

# 卷积块
class ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
        super(ConvBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)

# 上采样卷积块
class UpConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(UpConvBlock, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)

# Vision Mamba 编码器
class VisionMambaEncoder(nn.Module):
    def __init__(self, in_dim=3, depths=[1, 1, 1], dims=[16, 32, 64], d_state=8):
        super(VisionMambaEncoder, self).__init__()
        self.in_dim = in_dim
        self.depths = depths
        self.dims = dims
        self.d_state = d_state
        # 初始卷积层
        self.conv_in = ConvBlock(in_dim, dims[0])
        # 下采样和VSSBlock
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(len(depths)):
            if i > 0:
                self.downsamples.append(
                    nn.Sequential(
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(dims[i-1], dims[i], kernel_size=1, stride=1)
                    )
                )
            stage = nn.ModuleList()
            for j in range(depths[i]):
                stage.append(VSSBlock(
                    dim=dims[i],
                    d_state=d_state,
                    d_conv=3,
                    expand=2,
                    drop_path=0.0
                ))
            self.stages.append(stage)

    def forward(self, x):
        x = self.conv_in(x)
        features = [x]
        for i in range(len(self.depths)):
            if i > 0:
                x = self.downsamples[i-1](x)
            for block in self.stages[i]:
                x = block(x)
            features.append(x)
        return features

# Vision Mamba 解码器
class VisionMambaDecoder(nn.Module):
    def __init__(self, out_dim=1, depths=[1, 1], dims=[64, 32, 16], d_state=8):
        super(VisionMambaDecoder, self).__init__()
        self.out_dim = out_dim
        self.depths = depths
        self.dims = dims
        self.d_state = d_state
        self.stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in range(len(depths)):
            self.upsamples.append(
                UpConvBlock(dims[i], dims[i+1])
            )
            stage = nn.ModuleList()
            for j in range(depths[i]):
                stage.append(VSSBlock(
                    dim=dims[i+1],
                    d_state=d_state,
                    d_conv=3,
                    expand=2,
                    drop_path=0.0
                ))
            self.stages.append(stage)
        self.conv_out = nn.Sequential(
            nn.Conv2d(dims[-1], out_dim, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, features):
        x = features[-1]
        for i in range(len(self.depths)):
            x = self.upsamples[i](x)
            skip = features[-(i+2)]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([x, skip], dim=1)
            x = nn.Conv2d(x.shape[1], self.dims[i+1], kernel_size=1).to(x.device)(x)
            for block in self.stages[i]:
                x = block(x)
        output = self.conv_out(x)
        return output

# 完整的Vision Mamba UNet模型
class VisionMambaUNet(nn.Module):
    def __init__(self,
                 in_channels=6,
                 out_channels=1,
                 depths=[2, 2, 2],
                 dims=[16, 32, 64],
                 d_state=8,
                 **kwargs):
        super(VisionMambaUNet, self).__init__()
        self.encoder = VisionMambaEncoder(
            in_dim=in_channels,
            depths=depths,
            dims=dims,
            d_state=d_state
        )
        self.decoder = VisionMambaDecoder(
            out_dim=out_channels,
            depths=depths[:-1],
            dims=dims[::-1],
            d_state=d_state
        )
    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)
    def init_weights(self, pretrained=None, pretrained_transfer=None, strict=False, **kwargs):
        # 兼容原有权重加载逻辑
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            self.load_state_dict(new_dict, strict=strict)
        elif pretrained is None:
            def weights_init(m):
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            self.apply(weights_init)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.') 