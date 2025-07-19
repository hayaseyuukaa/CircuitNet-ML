# Copyright 2025 YourName. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv  # 使用关系图卷积
from collections import OrderedDict

def generation_init_weights(module):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1 or classname.find('RGCNConv') != -1):
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

class RGCNCongestion(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, hidden_channels=64, num_layers=2, 
                 num_relations=4, dropout=0.6, num_blocks=None, aggr='mean'):
        """
        RGCN模型用于拥塞预测
        
        Args:
            in_channels: 输入特征维度
            out_channels: 输出特征维度
            hidden_channels: 隐藏层维度
            num_layers: RGCN层数
            num_relations: 关系类型数量（4个方向：上下左右）
            dropout: dropout概率
            num_blocks: 正则化参数，None表示不使用块正则化
            aggr: 聚合方式 ('mean', 'add')
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.dropout = dropout
        
        # 构建多层RGCN
        self.rgcn_layers = nn.ModuleList()
        
        # 第一层
        self.rgcn_layers.append(RGCNConv(in_channels, hidden_channels, num_relations, 
                                       num_blocks=num_blocks, aggr=aggr))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.rgcn_layers.append(RGCNConv(hidden_channels, hidden_channels, num_relations,
                                           num_blocks=num_blocks, aggr=aggr))
        
        # 输出层
        if num_layers > 1:
            self.rgcn_layers.append(RGCNConv(hidden_channels, out_channels, num_relations,
                                           num_blocks=num_blocks, aggr=aggr))
        else:
            # 只有一层的情况
            self.rgcn_layers[0] = RGCNConv(in_channels, out_channels, num_relations,
                                         num_blocks=num_blocks, aggr=aggr)

    def forward(self, x):
        B, C, H, W = x.size()
        device = x.device
        
        # 展平为节点特征
        x = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)  # [B*H*W, in_channels]
        
        # 构建带关系类型的网格图邻接矩阵
        edge_index, edge_type = self._get_grid_edges_with_relations(B, H, W, device)
        
        # RGCN前向传播
        for i, rgcn_layer in enumerate(self.rgcn_layers):
            x = rgcn_layer(x, edge_index, edge_type)
            if i < len(self.rgcn_layers) - 1:  # 不在最后一层应用激活函数和dropout
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        
        # 重塑为图像格式
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)  # [B, out_channels, H, W]
        return torch.sigmoid(x)  # [B, 1, H, W]

    def _get_grid_edges_with_relations(self, batch_size, H, W, device):
        """
        构建带关系类型的网格图邻接关系
        关系类型定义：0=上, 1=下, 2=左, 3=右
        """
        num_nodes = H * W
        edges = []
        edge_relations = []
        
        for i in range(H):
            for j in range(W):
                node = i * W + j
                # 4邻域连接，每个方向对应一种关系类型
                if i > 0: 
                    edges.append([node, node - W])  # 上
                    edge_relations.append(0)  # 关系类型0：上
                if i < H - 1: 
                    edges.append([node, node + W])  # 下
                    edge_relations.append(1)  # 关系类型1：下
                if j > 0: 
                    edges.append([node, node - 1])  # 左
                    edge_relations.append(2)  # 关系类型2：左
                if j < W - 1: 
                    edges.append([node, node + 1])  # 右
                    edge_relations.append(3)  # 关系类型3：右
        
        single_edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()  # [2, num_edges]
        single_edge_type = torch.tensor(edge_relations, dtype=torch.long, device=device)  # [num_edges]
        
        # 为每个batch复制边和关系类型
        edge_index = []
        edge_type = []
        offset = torch.arange(batch_size, dtype=torch.long, device=device) * num_nodes  # [batch_size]
        
        for i in range(batch_size):
            edge_index.append(single_edge_index + offset[i])
            edge_type.append(single_edge_type)
        
        edge_index = torch.cat(edge_index, dim=1)  # [2, num_edges * batch_size]
        edge_type = torch.cat(edge_type, dim=0)    # [num_edges * batch_size]
        
        return edge_index, edge_type

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