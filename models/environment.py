"""
environment.py
──────────────
Microgrid MDP environment.

Implements Equations 1-5 (power balance, SoC dynamics, objective) and
the state/action/reward formulation of Equations 9-11.
"""

import numpy as np

ESS_CAP   = 500.0   # kWh
ESS_PMAX  = 100.0   # kW
ETA_CH    = 0.95
ETA_DIS   = 0.95
SOC_MIN   = 0.20
SOC_MAX   = 0.90
N_STEPS   = 96       # 15-min steps/day
DT        = 0.25     # hours

ALPHA, BETA, DELTA = 1.0, 0.5, 50.0   # reward weights (Eq. 11)


class MicrogridEnv:
    """
    Single-microgrid MDP environment (S, A, r, gamma).

    State  s_i,t (Eq. 9):  [GNN_embed | P_RES | P_load | SoC | price | P_grid]
    Action a_i,t (Eq. 10): [P_ch | P_dis | P_buy | P_sell | P_shed]  in [-1, 1]
    Reward r_i,t (Eq. 11): -alpha*Cost + beta*RES_util - delta*Violations
    """

    def __init__(self, mg_id: int, profiles: dict, gnn_dim: int = 64,
                 rng: np.random.Generator = None):
        self.mg_id   = mg_id
        self.sol48   = profiles['solar_mw'][mg_id]
        self.lod48   = profiles['load_mw'][mg_id]
        self.wnd48   = profiles['wind_mw'][mg_id]
        self.pb48    = profiles['price_buy']
        self.ps48    = profiles['price_sell']
        self.gnn_dim = gnn_dim
        self.rng     = rng or np.random.default_rng(mg_id)
        self.s_dim   = 6 + gnn_dim
        self.a_dim   = 5
        self.reset()

    def reset(self):
        self.t   = 0
        self.soc = 0.5 * ESS_CAP
        self._build_day()
        return self._state(np.zeros(self.gnn_dim, np.float32))

    def _build_day(self):
        """
        Resample 48-slot profiles to 96-step 15-min resolution and add
        stochastic perturbations (Eqs. 18-20):
            Solar: Beta-distributed noise
            Wind:  Weibull-distributed noise
            Load:  Gaussian noise (sigma=5%)
        """
        t48 = np.arange(48) * 0.5
        t96 = np.linspace(0, 24, N_STEPS, endpoint=False)

        sol = np.interp(t96, t48, self.sol48) * 1e3
        lod = np.interp(t96, t48, self.lod48) * 1e3
        wnd = np.interp(t96, t48, self.wnd48) * 1e3
        pb  = np.interp(t96, t48, self.pb48)
        ps  = np.interp(t96, t48, self.ps48)

        sol *= np.clip(self.rng.beta(2.0, 2.0, N_STEPS) * 0.3 + 0.85, 0, 1.5)
        wnd *= np.clip(self.rng.weibull(2.1, N_STEPS) * 0.8, 0.1, 2.0)
        lod *= np.clip(1.0 + self.rng.normal(0, 0.05, N_STEPS), 0.5, 1.5)

        self.sol_kw = np.clip(sol, 0, None)
        self.lod_kw = np.clip(lod, 0, None)
        self.wnd_kw = np.clip(wnd, 0, None)
        self.pb_day = pb
        self.ps_day = ps

    def _state(self, emb):
        t = min(self.t, N_STEPS - 1)
        s_peak = max(self.sol_kw.max(), 1e-6)
        l_peak = max(self.lod_kw.max(), 1e-6)
        local = np.array([
            self.sol_kw[t] / s_peak,
            self.lod_kw[t] / l_peak,
            self.soc / ESS_CAP,
            self.pb_day[t] / (self.pb_day.max() + 1e-9),
            0.0, 0.0
        ], np.float32)
        return np.concatenate([emb.astype(np.float32), local])

    def step(self, action, emb):
        """Execute one 15-min scheduling decision (Eqs. 1-5, 11)."""
        t = self.t
        if t >= N_STEPS:
            return self._state(emb), 0.0, True, {}

        p_ch   = float(np.clip((action[0]+1)/2 * ESS_PMAX, 0, ESS_PMAX))
        p_dis  = float(np.clip((action[1]+1)/2 * ESS_PMAX, 0, ESS_PMAX))
        p_buy  = float(np.clip((action[2]+1)/2 * 500, 0, 500))
        p_sell = float(np.clip((action[3]+1)/2 * 500, 0, 500))
        p_shed = float(np.clip((action[4]+1)/2 * self.lod_kw[t], 0, self.lod_kw[t]))

        p_res  = self.sol_kw[t] + self.wnd_kw[t]
        p_load = self.lod_kw[t] - p_shed

        net = p_res + p_buy - p_load - p_ch + p_dis - p_sell
        p_cur      = max(net, 0.0)
        p_grid_buy = max(-net, 0.0)

        soc_new = float(np.clip(
            self.soc + ETA_CH*p_ch*DT - (1/ETA_DIS)*p_dis*DT, 0, ESS_CAP))

        c_grid = (self.pb_day[t]*p_grid_buy - self.ps_day[t]*p_sell) * DT/1000
        c_op   = float(c_grid + 0.005*(p_ch+p_dis)*DT/1000
                       + 0.15*p_shed*DT/1000 + 0.05*p_cur*DT/1000)
        r_util = float((p_res - p_cur) / (p_res + 1e-9))

        sf  = soc_new / ESS_CAP
        vio = (max(0.0, sf-SOC_MAX) + max(0.0, SOC_MIN-sf)) * 100

        reward = -ALPHA*c_op + BETA*r_util - DELTA*vio

        self.soc = soc_new
        self.t  += 1
        done = self.t >= N_STEPS

        info = dict(cost=c_op, res_util=r_util, violation=vio,
                    soc_pct=soc_new/ESS_CAP*100, p_res_kw=p_res,
                    p_cur_kw=p_cur, p_grid_kw=p_grid_buy)
        return self._state(emb), reward, done, info


class MultiMGSystem:
    """Collection of N MicrogridEnv instances + shared GNN encoder."""

    def __init__(self, profiles: dict, n_mg: int = 6, gnn_dim: int = 64,
                 rng: np.random.Generator = None):
        from ieee33_microgrid.ieee33_topology import get_mg_adjacency

        self.n_mg = n_mg
        self.gnn_dim = gnn_dim
        adj_np = get_mg_adjacency(n_mg)

        try:
            import torch
            self.adj = torch.FloatTensor(adj_np)
            self._torch = True
        except ImportError:
            self.adj = adj_np
            self._torch = False

        self.envs = [
            MicrogridEnv(i, profiles, gnn_dim, np.random.default_rng(i*99+7))
            for i in range(n_mg)
        ]

        if self._torch:
            from models.gnn_encoder import MicrogridGNN
            self.gnn = MicrogridGNN(6, 64, gnn_dim, 2)

    def reset(self):
        return [e.reset() for e in self.envs]

    def local_features(self):
        return np.array([
            [e.sol_kw[min(e.t, N_STEPS-1)] / (e.sol_kw.max()+1e-9),
             e.lod_kw[min(e.t, N_STEPS-1)] / (e.lod_kw.max()+1e-9),
             e.soc / ESS_CAP,
             e.pb_day[min(e.t, N_STEPS-1)] / (e.pb_day.max()+1e-9),
             0.0, 0.0]
            for e in self.envs
        ], np.float32)

    def gnn_embed(self, feats):
        """Privacy-preserving GNN: share embeddings, not raw features (Sec. II-E)."""
        if not self._torch:
            return np.zeros((self.n_mg, self.gnn_dim), np.float32)
        import torch
        x = torch.FloatTensor(feats)
        return self.gnn(x, self.adj).detach().numpy()
