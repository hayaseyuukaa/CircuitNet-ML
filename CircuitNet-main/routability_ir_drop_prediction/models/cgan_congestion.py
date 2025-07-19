# Copyright 2025 Beifengdu. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, noise_dim=100):
        super().__init__()
        # 噪声输入
        self.noise_conv = nn.Conv2d(noise_dim, 64, kernel_size=1)

        # 编码器
        self.conv1 = nn.Conv2d(in_channels, 64, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(128, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)

        # 解码器
        self.upconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.upconv2 = nn.ConvTranspose2d(128, out_channels, 4, 2, 1)

    def forward(self, x, noise):
        # 噪声扩展为图像尺寸
        B = x.size(0)
        noise = noise.view(B, -1, 1, 1).expand(B, -1, x.size(2), x.size(3))  # [B, noise_dim, H, W]
        noise = F.relu(self.noise_conv(noise))  # [B, 64, H, W]

        # 编码
        x1 = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)  # [B, 64, H/2, W/2]
        x = torch.cat([x1, noise[:, :, ::2, ::2]], dim=1)  # [B, 128, H/2, W/2]
        x2 = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)  # [B, 128, H/4, W/4]

        # 解码
        x = F.relu(self.bn3(self.upconv1(x2)))  # [B, 64, H/2, W/2]
        x = torch.cat([x, x1], dim=1)  # [B, 128, H/2, W/2]
        x = self.upconv2(x)  # [B, 1, H, W]
        return torch.sigmoid(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, img_channels=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels + img_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0),
        )

    def forward(self, x, img):
        x = torch.cat([x, img], dim=1)  # [B, in_channels + img_channels, H, W]
        x = self.main(x)
        return torch.sigmoid(x.view(-1, 1))  # [B, 1]


class CGANCongestion(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, noise_dim=100):
        super().__init__()
        self.generator = Generator(in_channels, out_channels, noise_dim)
        self.discriminator = Discriminator(in_channels, out_channels)

    def forward(self, x, noise):
        # 生成器输出
        fake_img = self.generator(x, noise)
        return fake_img

    def discriminate(self, x, img):
        # 判别器输出
        return self.discriminator(x, img)

    def init_weights(self, pretrained=None, strict=True):
        if pretrained and isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')['state_dict']

            # 处理生成器的 state_dict
            new_generator_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('generator.'):
                    name = k[len('generator.'):]
                    new_generator_state_dict[name] = v
            self.generator.load_state_dict(new_generator_state_dict, strict=strict)

            # 处理判别器的 state_dict
            new_discriminator_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('discriminator.'):
                    name = k[len('discriminator.'):]
                    new_discriminator_state_dict[name] = v
            self.discriminator.load_state_dict(new_discriminator_state_dict, strict=strict)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, 0.0, 0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight, 1.0, 0.02)
                    nn.init.constant_(m.bias, 0)
