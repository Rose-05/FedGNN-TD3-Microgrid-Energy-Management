FedGNN-TD3: Federated Graph Neural Network and Twin Delayed Deep Reinforcement Learning for Distributed Renewable Energy Management

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Paper:** Rose Sadiki Lyimo, Pengfei Zhao, Weihao Hu — *FedGNN-TD3: A Federated Graph Neural Network and Twin Delayed Deep Reinforcement Learning Framework for Distributed Renewable Energy Management* — University of Electronic Science and Technology of China (UESTC), Chengdu, China.

---


Overview

FedGNN-TD3 is a privacy-preserving distributed energy management framework for interconnected multi-microgrid systems. It integrates three complementary technologies:

| Component | Role |
|---|---|
| **Graph Neural Network (GNN)** | Captures spatial dependencies and power exchange relationships among microgrids via topology-aware message passing |
| **Twin Delayed DDPG (TD3)** | Learns robust continuous control policies for ESS dispatch, renewable utilisation, and P2P energy trading under stochastic generation and load |
| **Federated Learning (FL)** | Enables collaborative policy improvement across microgrids without sharing raw operational data — preserving privacy |

The interconnected microgrid network is modelled as a graph of a modified IEEE 33-bus radial distribution system, partitioned into six microgrids, and evaluated using real-world renewable generation and load data from the Ausgrid and AEMO datasets.


Key Results

| Method | Cost (USD/day) | RES Util. (%) | Violation (%) |
|---|---|---|---|
| MILP (Centralised) | 1842.6 ± 0.0 | 87.4 ± 0.0 | 0.12 ± 0.00 |
| Centralised DRL | 1795.3 ± 24.7 | 91.2 ± 1.3 | 0.45 ± 0.08 |
| Multi-Agent DRL | 1768.9 ± 31.2 | 92.8 ± 1.5 | 0.38 ± 0.11 |
| GNN-MIP | 1725.4 ± 18.5 | 93.5 ± 0.9 | 0.25 ± 0.04 |
| **FedGNN-TD3 (Ours)** | **1658.7 ± 12.3** | **96.3 ± 0.6** | **0.09 ± 0.02** |

FedGNN-TD3 achieves a **9.98% cost reduction** vs MILP and **40.1% lower communication overhead** vs centralised data-sharing approaches.

Ablation study

| Variant | FL | GNN | Cost (USD/day) | RES Util. (%) | Violation (%) | Convergence (ep.) |
|---|---|---|---|---|---|---|
| DRL Only | No | No | 1823.4 | 88.6 | 0.52 | 142 |
| Fed-TD3 | Yes | No | 1768.1 | 91.4 | 0.38 | 118 |
| GNN-TD3 | No | Yes | 1712.6 | 93.9 | 0.21 | 95 |
| **FedGNN-TD3 (Full)** | Yes | Yes | **1658.7** | **96.3** | **0.09** | **72** |

---

Repository Structure

```
FedGNN-TD3-Microgrid-Energy-Management/
│
├── README.md                  ← This file
├── requirements.txt           ← Python dependencies
├── main.py                    ← Single entry point for training + evaluation
├── config.yaml                ← All hyperparameters and dataset paths
│
├── data/
│   ├── raw/                   ← Place downloaded datasets here (see data/README.md)
│   ├── processed/             ← Auto-generated preprocessed profiles
│   └── README.md              ← Dataset download instructions
│
├── ieee33_microgrid/
│   ├── ieee33_topology.py     ← IEEE 33-bus bus/branch/generator data
│   ├── microgrid_partition.py ← Six-MG partitioning and adjacency graph
│   └── data_mapping.py        ← Map dataset measurements onto bus topology
│
├── models/
│   ├── gnn_encoder.py         ← GNNLayer, MicrogridGNN (Eqs. 7–8)
│   ├── td3_agent.py           ← Actor, Critic, ReplayBuffer, TD3Agent (Eqs. 11–13)
│   └── fedavg.py              ← FedAvg aggregation (Eqs. 14–15)
│
├── training/
│   ├── train_local.py         ← Single-MG local TD3 training loop
│   ├── train_federated.py     ← Full FedGNN-TD3 Algorithm 1 loop
│   └── evaluate.py            ← Policy evaluation and metric computation (Eqs. 21–24)
│
├── results/
│   ├── figures/               ← Generated publication figures (PNG, 300 DPI)
│   └── tables/                ← Generated results tables (CSV)
│
├── scripts/
│   ├── preprocess_ausgrid.py  ← Load and preprocess all three datasets
│   ├── run_experiments.py     ← Run full ablation study and benchmark comparisons
│   └── plot_results.py        ← Reproduce all paper figures from saved results
│
└── docs/
    ├── architecture.png       ← Framework architecture diagram (placeholder — replace with Fig. 1)
    └── paper_summary.md       ← Methodology summary with equation references


## Datasets

Three real-world datasets are required. See [`data/README.md`](data/README.md) for download links and placement instructions.

| # | Dataset | Source | Used For |
|---|---|---|---|
| 1 | **Ausgrid Solar Home Electricity Data 2012–2013** | Ausgrid (Australia) | Solar PV generation (GG) and residential load (GC) for 300 customers, 30-min resolution |
| 2 | **AEMO 2018 NEM Dispatch** | Australian Energy Market Operator | Wind farm dispatch (33 farms), solar dispatch (28 farms), 5-min resolution → price proxy |
| 3 | **IEEE 33-Bus Benchmark** | Dolatabadi et al., IEEE Trans. Power Syst. 2020 | Bus loads, branch impedances, 24h demand/RES profiles, generator buses |

---

Installation

bash
Clone the repository
git clone https://github.com/YOUR_USERNAME/FedGNN-TD3-Microgrid-Energy-Management.git
cd FedGNN-TD3-Microgrid-Energy-Management

Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate         # Windows

Install dependencies
pip install -r requirements.txt


Quick Start

### 1. Download datasets
Follow instructions in [`data/README.md`](data/README.md) and place files in `data/raw/`.

### 2. Run with defaults (rule-based fallback, no PyTorch needed)
```bash
python main.py
```

### 3. Train full TD3 model (requires PyTorch)
```bash
python main.py --train --fed_rounds 50 --local_episodes 10 --seed 42
```

### 4. Evaluate and plot only
```bash
python main.py --evaluate --no_figures=False
```

All figures are written to `results/figures/` and tables to `results/tables/`.

---

## Configuration

All hyperparameters are in [`config.yaml`](config.yaml). Key settings:

```yaml
gnn:
  n_layers: 2
  hidden_dim: 64
  aggregation: mean

td3:
  actor_lr: 3.0e-4
  critic_lr: 3.0e-4
  gamma: 0.99
  policy_delay: 2        # delayed actor update (TD3 improvement #2)

federated:
  rounds: 50
  local_episodes: 10

reward:
  alpha: 1.0   # operational cost
  beta: 0.5    # RES utilisation
  delta: 50.0  # constraint violation penalty

