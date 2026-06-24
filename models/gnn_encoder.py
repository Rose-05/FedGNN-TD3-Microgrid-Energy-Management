"""
gnn_encoder.py
──────────────
Graph Neural Network for topology-aware feature extraction.

Implements Equations 7-8 from the paper:
    m_i^(l)   = AGGREGATE({ h_j^(l) : j in N(i) })          Eq. 7
    h_i^(l+1) = sigma( W^(l) [h_i^(l) || m_i^(l)] + b^(l) )  Eq. 8

Privacy-preserving message passing (Section II-E):
    Each MG sends its current embedding h_i^(l) — NOT raw features — to
    neighbours. This ensures sensitive operational data never leaves the
    local microgrid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNLayer(nn.Module):
    """
    Single message-passing layer implementing Eqs. 7-8.
    Aggregation: MEAN over neighbours (Table I: 'mean').
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)
        self.bn     = nn.BatchNorm1d(out_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        h   : (N, in_dim)   current node embeddings
        adj : (N, N)        adjacency matrix
        Returns (N, out_dim) updated node embeddings.
        """
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        m   = (adj @ h) / deg
        x   = torch.cat([h, m], dim=-1)
        return F.relu(self.bn(self.linear(x)))


class MicrogridGNN(nn.Module):
    """
    Multi-layer GNN for topology-aware state representation (Section II-B).

    Input feature x_i,t (Eq. 6): [P_RES, P_load, SoC, lambda, P_grid, P_tie_net]
    Output: node embedding h_i,t used in DRL state (Eq. 9)
    """

    def __init__(self, in_dim: int = 6, hidden_dim: int = 64,
                 embed_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GNNLayer(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.output_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_proj(x))
        for layer in self.layers:
            h = layer(h, adj)
        return self.output_proj(h)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_gnn(cfg: dict) -> MicrogridGNN:
    """Construct GNN from config dict."""
    gnn_cfg = cfg.get('gnn', {})
    return MicrogridGNN(
        in_dim=6,
        hidden_dim=gnn_cfg.get('hidden_dim', 64),
        embed_dim=gnn_cfg.get('embed_dim', 64),
        n_layers=gnn_cfg.get('n_layers', 2),
    )


if __name__ == '__main__':
    from ieee33_microgrid.ieee33_topology import get_mg_adjacency
    adj = torch.FloatTensor(get_mg_adjacency(6))
    x   = torch.randn(6, 6)
    gnn = MicrogridGNN()
    emb = gnn(x, adj)
    print(f"GNN output shape : {emb.shape}")
    print(f"Trainable params : {gnn.n_params:,}")
