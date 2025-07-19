# Copyright 2025 YourName. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # 替换为torch_geometric的GCNConv
from collections import OrderedDict

def generation_init_weights(module):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1 or classname.find('GATConv') != -1):
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
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True, all_missing_keys, unexpected_keys, err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    missing_keys = [key for key in all_missing_keys if 'num_batches_tracked' not in key]
    if unexpected_keys:
        err_msg.append(f'unexpected key in source state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(f'missing keys in source state_dict: {", ".join(missing_keys)}\n')
    if len(err_msg) > 0:
        err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    return missing_keys

class GCNCongestion(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, hidden_channels=64, dropout=0.6):
        super().__init__()
        self.gc1 = GCNConv(in_channels, hidden_channels)
        self.gc2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x):
        B, C, H, W = x.size()
        device = x.device  # 获取输入数据的设备
        
        # 展平为节点特征
        x = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)  # [B*H*W, in_channels]
        # 构建网格图邻接矩阵
        edge_index = self._get_grid_edges(B, H, W, device)  # [2, num_edges * B]

        # GCN前向传播
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)

        # 重塑为图像格式
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)  # [B, out_channels, H, W]
        return torch.sigmoid(x)  # [B, 1, H, W]

    def _get_grid_edges(self, batch_size, H, W, device):
        num_nodes = H * W
        edges = []
        for i in range(H):
            for j in range(W):
                node = i * W + j
                if i > 0: edges.append([node, node - W])
                if i < H - 1: edges.append([node, node + W])
                if j > 0: edges.append([node, node - 1])
                if j < W - 1: edges.append([node, node + 1])
        single_edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()  # [2, num_edges]
        edge_index = []
        offset = torch.arange(batch_size, dtype=torch.long, device=device) * num_nodes  # [batch_size]
        for i in range(batch_size):
            edge_index.append(single_edge_index + offset[i])
        edge_index = torch.cat(edge_index, dim=1)  # [2, num_edges * batch_size]
        return edge_index

    def init_weights(self, pretrained=None, strict=True, **kwargs):
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
        elif pretrained is None:
            generation_init_weights(self)
        else:
            raise TypeError("'pretrained' must be a str or None.")
    # def init_weights(self, pretrained=None, strict=True):
    #     if pretrained and isinstance(pretrained, str):
    #         state_dict = torch.load(pretrained, map_location='cpu')['state_dict']
    #         # 初始化权重（torch_geometric的GCNConv没有直接的weight属性，跳过预训练加载）
    #         nn.init.normal_(self.gc1.lin.weight, 0.0, 0.02)
    #         nn.init.normal_(self.gc2.lin.weight, 0.0, 0.02)