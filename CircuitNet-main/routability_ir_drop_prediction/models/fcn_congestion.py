# Copyright 2025 YourName. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNCongestion(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # 编码器
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 中间层
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # 解码器
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv5 = nn.Conv2d(512, 256, 3, padding=1)  # 跳跃连接后
        self.bn5 = nn.BatchNorm2d(256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv6 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv7 = nn.Conv2d(128, out_channels, 3, padding=1)

    def forward(self, x):
        # 编码器
        x1 = F.relu(self.bn1(self.conv1(x)))  # [B, 64, 256, 256]
        x2 = self.pool(F.relu(self.bn2(self.conv2(x1))))  # [B, 128, 128, 128]
        x3 = self.pool(F.relu(self.bn3(self.conv3(x2))))  # [B, 256, 64, 64]
        x4 = self.pool(F.relu(self.bn4(self.conv4(x3))))  # [B, 512, 32, 32]
        
        # 解码器
        x = self.upconv3(x4)  # [B, 256, 64, 64]
        x = torch.cat([x, x3], dim=1)  # [B, 512, 64, 64]
        x = F.relu(self.bn5(self.conv5(x)))  # [B, 256, 64, 64]
        x = self.upconv2(x)  # [B, 128, 128, 128]
        x = torch.cat([x, x2], dim=1)  # [B, 256, 128, 128]
        x = F.relu(self.bn6(self.conv6(x)))  # [B, 128, 128, 128]
        x = self.upconv1(x)  # [B, 64, 256, 256]
        x = torch.cat([x, x1], dim=1)  # [B, 128, 256, 256]
        x = self.conv7(x)  # [B, 1, 256, 256]
        return torch.sigmoid(x)

    def init_weights(self, pretrained=None, strict=True):
        if pretrained and isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')['state_dict']
            self.load_state_dict(state_dict, strict=strict)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, 0.0, 0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight, 1.0, 0.02)
                    nn.init.constant_(m.bias, 0)