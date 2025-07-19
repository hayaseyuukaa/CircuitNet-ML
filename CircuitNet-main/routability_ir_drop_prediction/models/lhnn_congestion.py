# # Copyright 2025 YourName. All rights reserved.
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class ResidualBlock(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.lin1 = nn.Linear(dim, dim)
#         self.lin2 = nn.Linear(dim, dim)
#
#     def forward(self, x):
#         x2 = F.relu(self.lin2(F.relu(self.lin1(x))))
#         return F.relu(x + x2)
#
#
#
# class LatticeMPBlock(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.res = ResidualBlock(dim)
#         self.lin = nn.Linear(dim, dim)
#         self.lin_a = nn.Linear(dim, dim)
#
#     def forward(self, v_c, na_cc):
#         v_c = self.res(v_c)
#         v1_c = F.relu(self.lin(v_c))
#         v2_c = F.relu(self.lin(na_cc @ v_c))
#         return v1_c + v2_c
#
#
# class FeatureGenBlock(nn.Module):
#     def __init__(self, n_dim, c_dim, dim):
#         super().__init__()
#         self.lin1_n = nn.Linear(n_dim, dim)
#         self.lin2_n = nn.Linear(dim, dim)
#         self.lin1_c = nn.Linear(c_dim, dim)
#         self.lin2_c = nn.Linear(2 * dim, dim)
#         self.res_n = ResidualBlock(dim)
#         self.res_c = ResidualBlock(dim)
#
#     def forward(self, v_n, v_c, g_nc):
#         v_n = self.res_n(F.relu(self.lin1_n(v_n)))
#         v_c = self.res_c(F.relu(self.lin1_c(v_c)))
#         v1_n = F.relu(self.lin2_n(v_n))
#         v1_c = F.relu(self.lin2_c(torch.cat([v_c, g_nc.t() @ v_n], dim=-1)))
#         return v1_n, v1_c
#
#
# class HyperMPBlock(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.lin1_n = nn.Linear(dim, dim)
#         self.lin2_n = nn.Linear(dim, dim)
#         self.lin3_n = nn.Linear(2 * dim, dim)
#         self.lin1_c = nn.Linear(dim, dim)
#         self.lin2_c = nn.Linear(dim, dim)
#         self.lin3_c = nn.Linear(2 * dim, dim)
#         self.res1_n = ResidualBlock(dim)
#         self.res2_n = ResidualBlock(dim)
#         self.res1_c = ResidualBlock(dim)
#         self.res2_c = ResidualBlock(dim)
#
#     def forward(self, v_n, v_c, v1_n, v1_c, g_nc, g_cn):
#         v_n = self.res1_n(v_n)
#         v_c = self.res1_c(v_c)
#         v_n = v_n + F.relu(self.lin3_n(torch.cat([F.relu(self.lin1_n(v1_n)), F.relu(self.lin2_n(g_cn.t() @ v_c))], dim=-1)))
#         v_n = self.res2_n(v_n)
#         v_c = v_c + F.relu(self.lin3_c(torch.cat([F.relu(self.lin1_c(v1_c)), F.relu(self.lin2_c(g_nc.t() @ v_n))], dim=-1)))
#         v_c = self.res2_c(v_c)
#         return v_n, v_c
#
#
# class LHNNCongestion(nn.Module):
#     def __init__(self, in_channels=3, out_channels=1, dim=256):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#
#         self.feature_gen = FeatureGenBlock(in_channels, in_channels, dim)
#         self.hyper_mp_1 = HyperMPBlock(dim)
#         self.hyper_mp_2 = HyperMPBlock(dim)
#         self.lattice_mp = LatticeMPBlock(dim)
#         self.lattice_mp_s1 = LatticeMPBlock(dim)
#         self.lattice_mp_s2 = LatticeMPBlock(dim)
#         self.readout = nn.Linear(dim, out_channels)
#
#     def forward(self, x):
#         B, C, H, W = x.size()
#         N = H * W  # 节点数
#
#         # 展平为节点特征
#         v_n = x.permute(0, 2, 3, 1).contiguous().view(B * N, C)  # [B*65536, 3]
#
#         # 构造超边（每16x16块一个超边，减少计算量）
#         block_size = 32
#         C_num = (H // block_size) * (W // block_size)  # 超边数，B=4时为256
#         v_c = torch.zeros(B * C_num, C, device=x.device)  # [B*256, 3]
#         g_nc = torch.zeros(B * N, B * C_num, device=x.device)  # [B*65536, B*256]
#
#         for b in range(B):
#             for i in range(0, H, block_size):
#                 for j in range(0, W, block_size):
#                     c_idx = b * C_num + (i // block_size) * (W // block_size) + (j // block_size)
#                     block = x[b, :, i:i + block_size, j:j + block_size].reshape(C, -1)
#                     v_c[c_idx] = block.mean(dim=1)
#                     n_start = b * N + i * W + j
#                     for ni in range(block_size):
#                         for nj in range(block_size):
#                             n_idx = n_start + ni * W + nj
#                             g_nc[n_idx, c_idx] = 1
#
#         g_cn = g_nc.t()  # [B*256, B*65536]
#         na_cc = g_cn @ g_nc  # [B*256, B*256]
#
#         # LHNN前向传播
#         v1_n, v1_c = self.feature_gen(v_n, v_c, g_nc)
#         v2_n, v2_c = self.hyper_mp_1(v1_n, v1_c, v1_n, v1_c, g_nc, g_cn)
#         v3_n, v3_c = self.hyper_mp_2(v2_n, v2_c, v1_n, v1_c, g_nc, g_cn)
#         v4_c = self.lattice_mp(v3_c, na_cc)
#         v5_c = self.lattice_mp_s1(v4_c, na_cc)
#         v6_c = self.lattice_mp_s2(v5_c, na_cc)
#         out = self.readout(v6_c)  # [B*256, 1]
#
#         # 重塑为图像格式
#         out = out.view(B, H // block_size, W // block_size, 1)  # [B, 16, 16, 1]
#         out = out.permute(0, 3, 1, 2)  # 转换为 [B, 1, 16, 16]
#         out = F.interpolate(out,
#                             scale_factor=block_size,  # 使用放大倍数更安全
#                             mode='bilinear',
#                             align_corners=False)  # [B, 1, 256, 256]
#         return torch.sigmoid(out)
#
#     def init_weights(self, pretrained=None, strict=True):
#         if pretrained and isinstance(pretrained, str):
#             state_dict = torch.load(pretrained, map_location='cpu')['state_dict']
#             self.load_state_dict(state_dict, strict=strict)
#         else:
#             for m in self.modules():
#                 if isinstance(m, nn.Linear):
#                     nn.init.normal_(m.weight, 0.0, 0.02)
#                     if m.bias is not None:
#                         nn.init.constant_(m.bias, 0)
#
# Copyright 2023 YourName. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 添加层归一化
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self._init_weights()

    def _init_weights(self):
        kaiming_normal_(self.lin1.weight, nonlinearity='leaky_relu')
        kaiming_normal_(self.lin2.weight, nonlinearity='leaky_relu')
        nn.init.zeros_(self.lin1.bias)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x):
        identity = x
        x = self.norm(x)  # 先归一化再线性层
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))
        return identity + x


class LatticeMPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.res = ResidualBlock(dim)
        self.norm = nn.LayerNorm(dim)  # 添加层归一化
        self.lin = nn.Linear(dim, dim)
        self.lin_a = nn.Linear(dim, dim)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, v_c, na_cc):
        v_c = self.res(v_c)
        v_c = self.norm(v_c)  # 归一化后传播
        v1_c = self.act(self.lin(v_c))
        v2_c = self.act(self.lin_a(na_cc @ v_c))
        return v1_c + v2_c


class FeatureGenBlock(nn.Module):
    def __init__(self, n_dim, c_dim, dim):
        super().__init__()
        self.norm_n = nn.LayerNorm(dim)  # 节点特征归一化
        self.norm_c = nn.LayerNorm(dim)  # 超边特征归一化

        self.lin_n = nn.Sequential(
            nn.Linear(n_dim, dim),
            nn.LeakyReLU(0.1),
            ResidualBlock(dim)
        )
        self.lin_c = nn.Sequential(
            nn.Linear(c_dim, dim),
            nn.LeakyReLU(0.1),
            ResidualBlock(dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.LayerNorm(dim),  # 融合后归一化
            nn.LeakyReLU(0.1),
            ResidualBlock(dim)
        )

    def forward(self, v_n, v_c, g_nc):
        v_n = self.norm_n(self.lin_n(v_n))
        v_c = self.norm_c(self.lin_c(v_c))
        fused = torch.cat([v_c, g_nc.t() @ v_n], dim=-1)
        return v_n, self.fusion(fused)


class HyperMPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm_n = nn.LayerNorm(dim)
        self.norm_c = nn.LayerNorm(dim)

        self.node_processor = nn.Sequential(
            ResidualBlock(dim),
            ResidualBlock(dim)
        )
        self.edge_processor = nn.Sequential(
            ResidualBlock(dim),
            ResidualBlock(dim)
        )
        self.fusion_n = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.1)
        )
        self.fusion_c = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.1)
        )

    def forward(self, v_n, v_c, v1_n, v1_c, g_nc, g_cn):
        # Node update
        node_feat = self.fusion_n(
            torch.cat([self.norm_n(v1_n),
                       self.norm_c(g_cn.t() @ v_c)], dim=-1)
        )
        v_n = self.node_processor(v_n + node_feat)

        # Edge update
        edge_feat = self.fusion_c(
            torch.cat([self.norm_c(v1_c),
                       self.norm_n(g_nc.t() @ v_n)], dim=-1)
        )
        v_c = self.edge_processor(v_c + edge_feat)

        return v_n, v_c


class LHNNCongestion(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dim=128, block_size=32):
        super().__init__()
        self.block_size = block_size

        # 初始化核心模块
        self.feature_gen = FeatureGenBlock(in_channels, in_channels, dim)
        self.hyper_mp_1 = HyperMPBlock(dim)
        self.hyper_mp_2 = HyperMPBlock(dim)
        self.lattice_mp = LatticeMPBlock(dim)
        self.lattice_mp_s1 = LatticeMPBlock(dim)
        self.lattice_mp_s2 = LatticeMPBlock(dim)

        # 输出层特殊初始化
        self.readout = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(dim // 2, out_channels)
        )
        self._init_weights()

    def _init_weights(self):
        # 最后一层微小初始化
        nn.init.normal_(self.readout[-1].weight, mean=0, std=1e-3)
        nn.init.constant_(self.readout[-1].bias, 0.0)

        # 其余层初始化
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.readout[-1]:
                kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _build_hypergraph(self, x):
        B, C, H, W = x.shape
        block_size = self.block_size
        assert H % block_size == 0 and W % block_size == 0

        # 节点特征 [B*H*W, C]
        v_n = x.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # 超边特征 [B*num_blocks, C]
        num_blocks = (H // block_size) * (W // block_size)
        v_c = F.unfold(x, block_size, stride=block_size)  # [B, C*block², num_blocks]
        v_c = v_c.view(B, C, -1, num_blocks).mean(dim=2)  # [B, C, num_blocks]
        v_c = v_c.permute(0, 2, 1).reshape(B * num_blocks, C)  # [B*num_blocks, C]

        # 高效构建关联矩阵
        idx_base = torch.arange(B, device=x.device)[:, None] * H * W
        block_idx = torch.arange(num_blocks, device=x.device).view(1, -1)
        g_nc = (idx_base + block_idx.repeat(B, 1)).view(-1)  # 超边全局索引

        return v_n, v_c, self._create_sparse_g_nc(B, H, W, block_size)

    def _create_sparse_g_nc(self, B, H, W, block_size):
        # 使用稀疏矩阵节省内存
        indices = []
        num_blocks = (H // block_size) * (W // block_size)

        for b in range(B):
            for c in range(num_blocks):
                i, j = divmod(c, W // block_size)
                h_start = i * block_size
                w_start = j * block_size
                for di in range(block_size):
                    for dj in range(block_size):
                        n_idx = b * H * W + (h_start + di) * W + (w_start + dj)
                        indices.append([n_idx, b * num_blocks + c])

        indices = torch.tensor(indices, dtype=torch.long).t()
        values = torch.ones(indices.shape, device=indices.device)

        return torch.sparse_coo_tensor(
            indices, values,
            size=(B * H * W, B * num_blocks)
        ).coalesce()

    def forward(self, x):
        B, C, H, W = x.size()

        # 构建超图（使用稀疏矩阵）
        v_n, v_c, g_nc = self._build_hypergraph(x)
        g_cn = g_nc.t()

        # 消息传递
        v1_n, v1_c = self.feature_gen(v_n, v_c, g_nc)
        v2_n, v2_c = self.hyper_mp_1(v1_n, v1_c, v1_n, v1_c, g_nc, g_cn)
        v3_n, v3_c = self.hyper_mp_2(v2_n, v2_c, v1_n, v1_c, g_nc, g_cn)

        # 格点处理
        na_cc = torch.sparse.mm(g_cn, g_nc.to_dense())  # 转为稠密计算
        v4_c = self.lattice_mp(v3_c, na_cc)
        v5_c = self.lattice_mp_s1(v4_c, na_cc)
        v6_c = self.lattice_mp_s2(v5_c, na_cc)

        # 输出预测（无Sigmoid）
        out = self.readout(v6_c)

        # 重塑输出形状
        num_blocks = (H // self.block_size) * (W // self.block_size)
        out = out.view(B, num_blocks, 1, 1)
        out = out.permute(0, 3, 1, 2)  # [B,1,num_blocks,1]
        out = F.interpolate(out,
                            size=(H, W),
                            mode='bilinear',
                            align_corners=False)
        return torch.sigmoid(out)  # BCEWithLogitsLoss需要原始logits

    def init_weights(self, pretrained=None, strict=True):
        if pretrained and isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')['state_dict']
            self.load_state_dict(state_dict, strict=strict)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0.0, 0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
# 训练时需要修改的地方：
# 1. 损失函数改为 BCEWithLogitsLoss
# 2. 添加梯度裁剪
# 3. 调整学习率到更小的值（如1e-4）
# 4. 确保输入数据已经归一化到合理范围