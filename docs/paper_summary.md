# FedGNN-TD3 — Methodology Summary

This document summarises the methodology of the paper and maps each
component to the corresponding code module.

---

## 1. System Model (Section II-A)

N interconnected microgrids, each with renewable generation, ESS, loads,
and tie-line connections. Discrete scheduling horizon T = 96 steps
(15-min resolution, 24 hours).

**Power balance (Eq. 1):**
```
P_RES + P_grid_buy + P_buy = P_load + P_dis + P_ch + P_sell + P_cur
```

**ESS SoC dynamics (Eq. 2):**
```
SoC[t+1] = SoC[t] + eta_ch * P_ch * dt - (1/eta_dis) * P_dis * dt
```
subject to `SoC_min <= SoC[t] <= SoC_max` (Eq. 3) and
`0 <= P_ch <= P_ch_max`, `0 <= P_dis <= P_dis_max` (Eq. 4).

**Objective (Eq. 5):** minimise total cost across all MGs and time steps:
```
min sum_t sum_i ( C_grid + C_ESS + C_cur + C_shed )
```

→ Implemented in: `models/environment.py` (`MicrogridEnv.step`)

---

## 2. Graph Representation (Section II-B)

Multi-MG system as undirected graph G = (V, E):
- V = microgrid nodes
- E = tie-line edges

**Node feature vector (Eq. 6):**
```
x_i,t = [P_RES, P_load, SoC, lambda, P_grid, P_tie_net]
```

**GNN aggregation (Eq. 7):**
```
m_i^(l) = AGGREGATE({ h_j^(l) : j in N(i) })     (mean aggregation)
```

**GNN update (Eq. 8):**
```
h_i^(l+1) = sigma( W^(l) [h_i^(l) || m_i^(l)] + b^(l) )
```

→ Implemented in: `ieee33_microgrid/microgrid_partition.py`,
  `models/gnn_encoder.py`

---

## 3. TD3-Based DRL Formulation (Section II-C)

MDP tuple (S, A, r, gamma). TD3 chosen over DDPG for three improvements:
1. Clipped double-Q learning — reduces overestimation bias
2. Delayed policy updates — reduces variance
3. Target policy smoothing — reduces variance

**State (Eq. 9):**
```
s_i,t = [h_i,t (GNN embed), P_RES, P_load, SoC, lambda, P_grid_buy]
```

**Action (Eq. 10):**
```
a_i,t = [P_ch, P_dis, P_buy, P_sell, P_shed]
```

**Reward (Eq. 11):**
```
r_i,t = -alpha * C_i,t + beta * R_i,t - delta * V_i,t
```

**Objective (Eq. 12):**
```
J(pi_i) = E[ sum_t gamma^(t-1) * r_i,t ]
```

**Deterministic policy gradient (Eq. 13):**
```
grad_theta J ~ E_s [ grad_a Q_phi1(s,a)|a=pi_theta(s) * grad_theta pi_theta(s) ]
```

→ Implemented in: `models/td3_agent.py`, `models/environment.py`

---

## 4. Federated Learning Integration (Section II-D)

**FedAvg aggregation (Eq. 14):**
```
theta_global^(r+1) = sum_i (n_i / sum_j n_j) * theta_i^r
```

**Broadcast (Eq. 15):**
```
theta_i^(r+1) <- theta_global^(r+1),  for all i
```

→ Implemented in: `models/fedavg.py`, `training/train_federated.py`

---

## 5. Privacy-Preserving GNN Message Passing (Section II-E)

Each microgrid computes its own node embedding `h_i^(l)` using only local
data and previously received **anonymised embeddings** from neighbours.
Raw feature vectors are never transmitted between microgrids — only
embeddings (after the GNN forward pass).

→ Implemented in: `models/environment.py`
  (`MultiMGSystem.gnn_embed` shares embeddings, not `local_features`)

---

## 6. Computational Complexity (Section II-F)

Per federated round complexity:
```
O( N * (T * E_TD3 + L * H^2 * |E|) )       (local computation, Eq. 16)
O( N * P )                                  (FedAvg overhead, Eq. 17)
```
where N = #MGs, T = time steps, L = GNN layers, H = hidden dim,
|E| = #graph edges, P = #model parameters.

---

## 7. Algorithm 1 — Full Training Loop

```
Require: N microgrids, R federated rounds, K local episodes, graph G
Ensure: Optimised global policy pi_global

1:  Initialise theta_i for each MG, and theta_global
2:  Construct microgrid graph G from tie-line connections
3:  for r = 1 to R:
4:    for i = 1 to N in parallel:
5:      theta_i <- theta_global                (download)
6:      for k = 1 to K:
7:        Run local TD3 episode with privacy-preserving GNN
8:        Update theta_i using stored transitions   (local training)
9:      Send delta_theta_i = theta_i - theta_global to server
10:   Server: theta_global <- FedAvg(theta_1, ..., theta_N)
11:   Broadcast theta_global to all microgrids
12: return pi_global
```

→ Implemented in: `training/train_federated.py` (`run_fedgnn_td3`)

---

## 8. Evaluation Metrics (Section III-E)

**Total operational cost (Eq. 21):**
```
C_tot = sum_t ( C_grid_t + C_ESS_t + C_trade_t )
```

**Renewable utilisation rate (Eq. 22):**
```
eta_RES = sum(P_RES_used) / sum(P_RES_avail) * 100%
```

**Constraint violation rate (Eq. 23):**
```
V_rate = N_viol / N_total * 100%
```

**Training time (Eq. 24):**
```
T_tr
