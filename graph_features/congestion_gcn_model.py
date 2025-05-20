import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv


class CongestionGCN(nn.Module):
    """基于图卷积网络的拥塞预测模型"""

    def __init__(
        self,
        in_dim=12,
        hidden_dim=128,
        out_dim=2,
        num_layers=3,
        dropout=0.2,
        use_sage=True,
        use_gat=False,
        residual=True,
        norm=True,
    ):
        """
        参数:
            in_dim: 输入特征维度，默认为12（带拥塞特征）或10（不带拥塞特征）
            hidden_dim: 隐藏层维度
            out_dim: 输出维度，预测水平和垂直方向的拥塞度，所以是2
            num_layers: GCN层数
            dropout: Dropout比率
            use_sage: 是否使用GraphSAGE代替普通GCN
            use_gat: 是否使用GAT代替普通GCN
            residual: 是否使用残差连接
            norm: 是否使用批归一化
        """
        super(CongestionGCN, self).__init__()
        self.num_layers = num_layers
        self.residual = residual
        self.norm = norm

        # 输入层
        self.embedding = nn.Linear(in_dim, hidden_dim)

        # 图卷积层
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            if use_gat:
                # 图注意力层
                self.layers.append(
                    GATConv(
                        hidden_dim,
                        hidden_dim // 4,
                        4,
                        feat_drop=dropout,
                        attn_drop=dropout,
                        residual=residual,
                    )
                )
            elif use_sage:
                # GraphSAGE层
                self.layers.append(SAGEConv(hidden_dim, hidden_dim, "mean"))
            else:
                # 普通GCN层
                self.layers.append(
                    GraphConv(
                        hidden_dim, hidden_dim, norm="both", weight=True, bias=True
                    )
                )

            if norm:
                self.norms.append(nn.BatchNorm1d(hidden_dim))

        # 输出层
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

        # 边权重转换层
        self.edge_weight_fc = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())

    def forward(self, g, features, edge_weights=None):
        """
        前向传播

        参数:
            g: DGL图
            features: 节点特征
            edge_weights: 边权重特征

        返回:
            输出特征，表示节点的拥塞预测结果 [水平拥塞, 垂直拥塞]
        """
        h = self.embedding(features)

        # 处理边权重
        if edge_weights is not None:
            edge_weights = self.edge_weight_fc(edge_weights)

        h_last = h  # 用于残差连接

        for i in range(self.num_layers):
            if edge_weights is not None:
                h = self.layers[i](g, h, edge_weight=edge_weights)
            else:
                h = self.layers[i](g, h)

            if self.norm:
                h = self.norms[i](h)

            h = F.relu(h)
            h = F.dropout(h, p=0.2, training=self.training)

            if self.residual and i > 0:
                h = h + h_last

            h_last = h

        # 输出层
        out = self.fc_out(h)
        return out


class CongestionGAT(nn.Module):
    """基于图注意力网络的拥塞预测模型"""

    def __init__(
        self,
        in_dim=12,
        hidden_dim=128,
        out_dim=2,
        num_layers=3,
        num_heads=4,
        dropout=0.2,
    ):
        super(CongestionGAT, self).__init__()
        self.num_layers = num_layers

        # 输入层
        self.embedding = nn.Linear(in_dim, hidden_dim)

        # 图注意力层
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            # Multi-head attention
            if i < num_layers - 1:
                self.layers.append(
                    GATConv(
                        hidden_dim,
                        hidden_dim // num_heads,
                        num_heads,
                        feat_drop=dropout,
                        attn_drop=dropout,
                        residual=True,
                    )
                )
            else:
                # 最后一层，合并多头注意力
                self.layers.append(
                    GATConv(
                        hidden_dim,
                        hidden_dim // num_heads,
                        num_heads,
                        feat_drop=dropout,
                        attn_drop=dropout,
                        residual=True,
                    )
                )

            self.norms.append(nn.BatchNorm1d(hidden_dim))

        # 输出层
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, g, features):
        h = self.embedding(features)

        for i in range(self.num_layers):
            h = self.layers[i](g, h)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=0.2, training=self.training)

        out = self.fc_out(h)
        return out


def build_model(model_type="gcn", in_dim=12, hidden_dim=128, out_dim=2, **kwargs):
    """构建模型工厂函数

    参数:
        model_type: 模型类型, 'gcn', 'gat', 'sage'
        in_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        out_dim: 输出维度
        **kwargs: 其他参数

    返回:
        实例化的模型
    """
    if model_type == "gat":
        return CongestionGAT(in_dim, hidden_dim, out_dim, **kwargs)
    elif model_type == "sage":
        return CongestionGCN(
            in_dim, hidden_dim, out_dim, use_sage=True, use_gat=False, **kwargs
        )
    else:  # 默认gcn
        return CongestionGCN(
            in_dim, hidden_dim, out_dim, use_sage=False, use_gat=False, **kwargs
        )
