import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class DenoisingLayer(nn.Module):
    """动态去噪层，可以调整噪声水平并学习去噪"""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x, noise_level=0.0):
        """添加噪声并使用残差连接去除噪声"""
        # 根据噪声水平添加噪声
        if noise_level > 0 and self.training:
            noise = torch.randn_like(x) * noise_level
            x_noisy = x + noise
        else:
            x_noisy = x

        # 去噪网络
        residual = x_noisy
        out = F.relu(self.bn1(self.conv1(x_noisy)))
        out = self.bn2(self.conv2(out))

        # 残差连接
        return F.relu(out + residual)


class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=100):
        super().__init__()
        # 动态去噪层
        self.denoise = DenoisingLayer(in_channels)

        # 编码器
        self.conv1 = nn.Conv2d(in_channels, 64, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(256)

        # 计算输入图像经过三次下采样后的特征图尺寸
        # 假设输入为 H x W，则输出为 H/8 x W/8
        self.fc_mu = nn.Conv2d(256, latent_dim, 1)
        self.fc_logvar = nn.Conv2d(256, latent_dim, 1)

    def forward(self, x, noise_level=0.0):
        # 动态去噪
        x = self.denoise(x, noise_level)

        # 编码
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)  # [B, 64, H/2, W/2]
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)  # [B, 128, H/4, W/4]
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)  # [B, 256, H/8, W/8]

        # 生成均值和对数方差
        mu = self.fc_mu(x)  # [B, latent_dim, H/8, W/8]
        logvar = self.fc_logvar(x)  # [B, latent_dim, H/8, W/8]

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=100, out_channels=1):
        super().__init__()
        # 解码器
        self.conv1 = nn.Conv2d(latent_dim + in_channels, 256, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.upconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.upconv3 = nn.ConvTranspose2d(64, out_channels, 4, 2, 1)

        # 输出去噪层
        self.output_denoise = DenoisingLayer(out_channels)

    def forward(self, z, condition, noise_level=0.0):
        # 检查输入张量
        if z is None or condition is None:
            print("警告: 输入为None")
            return torch.zeros((condition.size(0), self.upconv3.out_channels, condition.size(2), condition.size(3)), 
                              device=condition.device)
        
        # 检查输入张量维度
        if condition.dim() != 4:
            print(f"警告: condition应为4D张量，但得到{condition.dim()}D")
            return torch.zeros((condition.size(0), self.upconv3.out_channels, 256, 256), 
                              device=condition.device)
        
        # 检查形状是否包含0
        if 0 in condition.size():
            print(f"警告: condition形状含0: {condition.size()}")
            # 创建新的有效条件张量
            valid_shape = list(condition.size())
            for i in range(len(valid_shape)):
                if valid_shape[i] == 0:
                    if i == 2:  # 高度
                        valid_shape[i] = 256
                    elif i == 3:  # 宽度
                        valid_shape[i] = 256
            condition = torch.zeros(valid_shape, device=condition.device)
            print(f"修复后的condition形状: {condition.size()}")
        
        # 保存原始条件图像的尺寸，用于确保输出尺寸匹配
        original_size = (condition.size(2), condition.size(3))

        # 确保z和condition有相同的空间维度
        try:
            if z.size(2) != condition.size(2) or z.size(3) != condition.size(3):
                z = F.interpolate(z, size=(condition.size(2), condition.size(3)), mode='bilinear', align_corners=False)
        except RuntimeError as e:
            print(f"插值错误: {e}")
            print(f"z形状: {z.size()}, condition形状: {condition.size()}")
            # 创建有效的z张量
            z = torch.randn(condition.size(0), z.size(1), condition.size(2), condition.size(3), device=condition.device)

        # 将隐变量和条件连接
        x = torch.cat([z, condition], dim=1)  # [B, latent_dim+in_channels, H, W]

        # 解码
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.upconv1(x)))
        x = F.relu(self.bn3(self.upconv2(x)))
        x = self.upconv3(x)

        # 确保输出与原始条件图像尺寸匹配
        current_size = (x.size(2), x.size(3))
        if current_size != original_size:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)

        # 输出去噪（可选，实际训练中可能不需要）
        x = torch.sigmoid(x)
        x = self.output_denoise(x, noise_level)

        return x


class CVAECongestion(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, latent_dim=100):
        super().__init__()
        self.encoder = Encoder(in_channels + out_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim, out_channels)
        self.latent_dim = latent_dim
        self.current_noise_level = 0.1  # 初始噪声水平

    def set_noise_level(self, noise_level):
        """设置当前噪声水平"""
        self.current_noise_level = noise_level

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, img=None, sampling=True, noise_level=None):
        # 如果没有指定噪声水平，则使用当前设置的水平
        if noise_level is None:
            noise_level = self.current_noise_level if self.training else 0.0

        # 检查输入
        if x is None:
            print("错误: 输入x为None")
            return torch.zeros((1, self.decoder.upconv3.out_channels, 256, 256), device='cuda' if torch.cuda.is_available() else 'cpu'), None, None
            
        # 检查输入维度
        if x.dim() != 4:
            print(f"警告: 输入x应为4D张量，但得到{x.dim()}D")
            # 尝试reshape为4D
            try:
                if x.dim() == 3:
                    x = x.unsqueeze(0)  # 添加批次维度
                elif x.dim() < 3:
                    print("错误: 输入维度过低，无法处理")
                    return torch.zeros((1, self.decoder.upconv3.out_channels, 256, 256), device=x.device), None, None
            except Exception as e:
                print(f"处理输入维度错误: {e}")
                return torch.zeros((1, self.decoder.upconv3.out_channels, 256, 256), device=x.device), None, None
                
        # 检查形状是否包含0
        if 0 in x.size():
            print(f"警告: 输入x形状含0: {x.size()}")
            # 创建新的有效输入张量
            valid_shape = list(x.size())
            for i in range(len(valid_shape)):
                if valid_shape[i] == 0:
                    if i == 2:  # 高度
                        valid_shape[i] = 256
                    elif i == 3:  # 宽度
                        valid_shape[i] = 256
            new_x = torch.zeros(valid_shape, device=x.device)
            if x.size(1) > 0:
                # 复制有效通道
                valid_channels = min(x.size(1), valid_shape[1])
                for c in range(valid_channels):
                    new_x[:, c].fill_(0.5)  # 用中间值填充
            x = new_x
            print(f"修复后的x形状: {x.size()}")

        # 保存输入尺寸，用于确保输出尺寸匹配
        if img is not None:
            original_size = (img.size(2), img.size(3))
        else:
            original_size = (x.size(2), x.size(3))

        # 训练时，img是真实标签；推理时，img为None
        if img is not None:
            # 编码器输入包括条件(x)和目标图像(img)
            try:
                encoder_input = torch.cat([x, img], dim=1)
                # 得到隐空间分布，传递噪声水平
                mu, logvar = self.encoder(encoder_input, noise_level)
                # 采样隐变量
                if sampling:
                    z = self.reparameterize(mu, logvar)
                else:
                    z = mu
            except Exception as e:
                print(f"编码器处理错误: {e}")
                # 直接从标准正态分布采样
                B, _, H, W = x.size()
                h, w = max(H // 8, 1), max(W // 8, 1)  # 确保至少为1
                z = torch.randn(B, self.latent_dim, h, w, device=x.device)
                mu, logvar = None, None
        else:
            # 推理时，没有目标图像，直接从标准正态分布采样
            B, _, H, W = x.size()
            # 计算下采样后的特征图大小，确保至少为1
            h, w = max(H // 8, 1), max(W // 8, 1)
            z = torch.randn(B, self.latent_dim, h, w, device=x.device)
            mu, logvar = None, None

        # 解码生成图像，传递噪声水平
        try:
            output = self.decoder(z, x, noise_level)
            
            # 最终检查，确保输出尺寸与原始图像完全匹配
            if output.size(2) != original_size[0] or output.size(3) != original_size[1]:
                output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)
                
            return output, mu, logvar
        except Exception as e:
            print(f"解码器处理错误: {e}")
            # 返回空结果
            return torch.zeros((x.size(0), self.decoder.upconv3.out_channels, original_size[0], original_size[1]), 
                              device=x.device), mu, logvar

    def generate_samples(self, x, num_samples=1, noise_scale=1.0):
        """生成多个样本，可控制潜在空间采样的多样性"""
        B, _, H, W = x.size()
        h, w = H // 8, W // 8
        samples = []

        # 保存原始图像尺寸
        original_size = (x.size(2), x.size(3))

        for i in range(num_samples):
            # 使用不同的噪声尺度生成多样化样本
            z = torch.randn(B, self.latent_dim, h, w, device=x.device) * noise_scale
            with torch.no_grad():
                # 解码生成样本
                sample = self.decoder(z, x, 0.0)

                # 确保生成的样本与原始图像尺寸匹配
                if sample.size(2) != original_size[0] or sample.size(3) != original_size[1]:
                    sample = F.interpolate(sample, size=original_size, mode='bilinear', align_corners=False)

                samples.append(sample)

        return samples

    def interpolate_samples(self, x, num_steps=10):
        """生成两个随机潜在向量之间的插值样本"""
        B, _, H, W = x.size()
        h, w = H // 8, W // 8
        original_size = (x.size(2), x.size(3))

        # 生成两个随机潜在向量
        z1 = torch.randn(B, self.latent_dim, h, w, device=x.device)
        z2 = torch.randn(B, self.latent_dim, h, w, device=x.device)

        samples = []
        # 在两个潜在向量之间进行线性插值
        for t in torch.linspace(0, 1, num_steps):
            # 线性插值
            z_t = z1 * (1 - t) + z2 * t

            with torch.no_grad():
                # 解码生成样本
                sample = self.decoder(z_t, x, 0.0)

                # 确保生成的样本与原始图像尺寸匹配
                if sample.size(2) != original_size[0] or sample.size(3) != original_size[1]:
                    sample = F.interpolate(sample, size=original_size, mode='bilinear', align_corners=False)

                samples.append(sample)

        return samples

    def init_weights(self, pretrained=None, strict=True):
        if pretrained and isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')['state_dict']

            # 处理编码器的 state_dict
            new_encoder_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('encoder.'):
                    name = k[len('encoder.'):]
                    new_encoder_state_dict[name] = v
            self.encoder.load_state_dict(new_encoder_state_dict, strict=strict)

            # 处理解码器的 state_dict
            new_decoder_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('decoder.'):
                    name = k[len('decoder.'):]
                    new_decoder_state_dict[name] = v
            self.decoder.load_state_dict(new_decoder_state_dict, strict=strict)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, 0.0, 0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight, 1.0, 0.02)
                    nn.init.constant_(m.bias, 0)
