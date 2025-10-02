import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_max_pool
from einops import rearrange


class GNNFeatureExtractor(nn.Module):
    def __init__(self, node_dim=11, hidden_dim=256, output_dim=768):
        super().__init__()
        # 图注意力层
        self.conv1 = GATConv(node_dim, hidden_dim, heads=4)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=2)
        # 输出投影到UNet所需维度
        self.proj = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # GNN处理
        x = F.relu(self.conv1(x, edge_index))  # [n, hidden_dim*4]
        x = F.relu(self.conv2(x, edge_index))  # [n, hidden_dim*2]

        # 全局特征聚合 (可选方案1)
        global_feat = global_max_pool(x, batch=None)  # [1, hidden_dim*2]

        # 投影到UNet维度
        return self.proj(global_feat)  # [1, output_dim]


class PointCloudTokenizer(nn.Module):
    def __init__(self, node_dim=11, hidden_dim=256, num_tokens=16):
        super().__init__()
        self.token_proj = nn.Linear(node_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, hidden_dim))
        self.pos_embed = nn.Linear(3, hidden_dim)  # 使用XYZ坐标作位置编码

    def forward(self, data):
        # 点特征投影
        x = self.token_proj(data.x)  # [n, hidden_dim]

        # 添加坐标位置编码
        pos_enc = self.pos_embed(data.pos)  # [n, hidden_dim]
        x = x + pos_enc

        # 添加可学习的CLS token
        cls_tokens = self.cls_token.expand(1, -1)  # [1, hidden_dim]
        x = torch.cat([cls_tokens, x], dim=0)  # [n+1, hidden_dim]

        return x  # 输出可直接作为Cross-Attention的key/value

