"""
microgrid_partition.py
──────────────────────
Defines the six-MG partitioning of the IEEE 33-bus system and provides
the MicrogridGraph data structure used by the GNN message-passing layer.
"""

import numpy as np
from ieee33_microgrid.ieee33_topology import get_mg_adjacency

MG_BUS = {
    0: list(range(1,  7)),
    1: list(range(7,  12)),
    2: list(range(12, 18)),
    3: list(range(18, 23)),
    4: list(range(23, 28)),
    5: list(range(28, 34)),
}

TIE_LINES = [(8, 21), (12, 22), (25, 29)]


class MicrogridGraph:
    """
    Graph G = (V, E) representing the multi-MG network (Sec. II-B).

    Nodes V : microgrids (each node = one MG)
    Edges E : tie-line connections between adjacent MGs

    Node feature vector x_i,t (Eq. 6):
        [P_RES, P_load, SoC, lambda, P_grid, P_tie_net]
    """

    def __init__(self, n_mg: int = 6):
        self.n_mg = n_mg
        self.adj  = get_mg_adjacency(n_mg)
        self.edges = self._build_edge_list()
        self.feature_dim = 6

    def _build_edge_list(self):
        edges = []
        for i in range(self.n_mg):
            for j in range(self.n_mg):
                if self.adj[i, j] > 0:
                    edges.append((i, j))
        return edges

    def neighbours(self, mg_id: int):
        return [j for j in range(self.n_mg) if self.adj[mg_id, j] > 0]

    def node_features(self, p_res, p_load, soc, price, p_grid, p_tie) -> np.ndarray:
        """Build normalised feature vector x_i,t for a single node (Eq. 6)."""
        return np.array([p_res, p_load, soc, price, p_grid, p_tie],
                        dtype=np.float32)

    def summary(self):
        print(f"  MicrogridGraph: {self.n_mg} nodes, {len(self.edges)} directed edges")
        for mg_id in range(self.n_mg):
            print(f"    MG {mg_id+1} (buses {MG_BUS[mg_id][0]}-{MG_BUS[mg_id][-1]})"
                  f"  <->  MGs {[n+1 for n in self.neighbours(mg_id)]}")


if __name__ == '__main__':
    g = MicrogridGraph()
    g.summary()
    print(f"\nAdjacency matrix:\n{g.adj}")
