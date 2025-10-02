import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool


def build_graph_from_gaussians(k=8):
    # xyz = gaussians['xyz']                    # (n, 3)
    # features = gaussians['features'].squeeze(1)  # (n, 3)
    # opacity = gaussians['opacity']            # (n, 1)
    # rotation = gaussians['rotation']          # (n, 4)
    # scaling = gaussians['scaling']            # (n, 3)
    xyz = torch.load("data_3d/fox/xyz.pt")
    features = torch.load("data_3d/fox/features.pt").squeeze(1)
    opacity = torch.load("data_3d/fox/opacity.pt")
    rotation = torch.load("data_3d/fox/rotation.pt")
    scaling = torch.load("data_3d/fox/scaling.pt")

    # 节点特征: (n, 14)
    # x = torch.cat([xyz, features, opacity, rotation, scaling], dim=1)
    xyz = torch.randn((100, 3))
    x = torch.randn((100, 14))
    # 构建 k-NN 图结构
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(xyz.detach().cpu().numpy())
    knn_idx = nbrs.kneighbors(return_distance=False)[:, 1:]  # (n, k)

    # 生成 edge_index (2, E)
    src = torch.arange(xyz.size(0)).repeat_interleave(k)
    dst = torch.tensor(knn_idx.flatten(), dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    return Data(x=x, edge_index=edge_index)


class GaussianGraphEncoder(nn.Module):
    def __init__(self, in_dim=14, hidden_dim=768, out_dim=1024):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        # 默认只有一个图（batch=0）
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_feat = global_max_pool(x, batch)  # shape: (1, out_dim)
        return graph_feat.squeeze(0)  # shape: (out_dim,)


gaussian_graph = build_graph_from_gaussians().to("cuda")
encoder = GaussianGraphEncoder().to("cuda")

with torch.no_grad():
    feature_vector = encoder(gaussian_graph)  # shape: (256,)
print("Graph feature vector shape:", feature_vector.shape)
