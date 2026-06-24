"""
run_experiments.py
───────────────────
Run the full ablation study (Table III) and benchmark comparison (Table II).

Ablation variants:
    DRL Only           — no Federated Learning, no GNN
    Fed-TD3            — Federated Learning only
    GNN-TD3            — GNN only
    FedGNN-TD3 (Full)  — both FL and GNN (proposed method)

Usage
──────
    python scripts/run_experiments.py --config config.yaml --train
    python scripts/run_experiments.py --config config.yaml          # rule-based
"""

import argparse
import pathlib
import sys
import copy

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

from scripts.preprocess_ausgrid import preprocess_all
from models.environment import MultiMGSystem
from training.evaluate import (evaluate_policy, action_rule_greedy,
                               action_peak_shift, action_no_storage)

try:
    import torch
    from models.td3_agent import TD3Agent
    from models.fedavg import federated_average
    TORCH = True
except ImportError:
    TORCH = False


def parse_args():
    p = argparse.ArgumentParser(description='Run FedGNN-TD3 ablation study')
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--train', action='store_true',
                   help='Run full TD3 training for each ablation variant '
                        '(requires PyTorch; slow)')
    p.add_argument('--output_dir', default=None)
    return p.parse_args()


def train_variant(profiles, cfg, use_fl: bool, use_gnn: bool,
                  n_episodes_per_round: int = None):
    """
    Train one ablation variant.

    use_fl  : if True, aggregate across MGs with FedAvg each round.
              if False, each MG trains independently (no aggregation).
    use_gnn : if True, use GNN embeddings as part of the state.
              if False, use a zero vector instead (no spatial information).
    """
    if not TORCH:
        return None

    n_mg       = cfg['system']['n_mg']
    fed_rounds = cfg['federated']['rounds']
    local_eps  = cfg['federated']['local_episodes']
    seed       = cfg['training']['seed']
    expl_noise = cfg['td3']['expl_noise']
    gnn_dim    = cfg['gnn']['embed_dim']

    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    system = MultiMGSystem(profiles, n_mg=n_mg, gnn_dim=gnn_dim, rng=rng)
    sd, ad = system.envs[0].s_dim, system.envs[0].a_dim

    agents   = [TD3Agent(sd, ad, cfg=cfg['td3']) for _ in range(n_mg)]
    global_p = agents[0].get_params()

    reward_history = []

    for r in range(fed_rounds):
        local_ps, n_samples = [], []
        round_rewards = []

        for i, (agent, env) in enumerate(zip(agents, system.envs)):
            if use_fl:
                agent.set_params(global_p)   # download global model

            for _ in range(local_eps):
                s = env.reset()
                ep_reward = 0.0
                while True:
                    if use_gnn:
                        feats = system.local_features()
                        emb   = system.gnn_embed(feats)[i]
                    else:
                        emb = np.zeros(gnn_dim, np.float32)
                    s[:len(emb)] = emb
                    a = agent.select_action(s, noise=expl_noise)
                    s_, reward, done, info = env.step(a, emb)
                    agent.replay_buffer.add(s, a, reward, s_, float(done))
                    s = s_; ep_reward += reward
                    agent.train_step()
                    if done:
                        break
                round_rewards.append(ep_reward)

            if use_fl:
                local_ps.append(agent.get_params())
                n_samples.append(len(agent.replay_buffer))

        if use_fl:
            global_p = federated_average(local_ps, n_samples)

        reward_history.append(float(np.mean(round_rewards)))

    return dict(agents=agents, global_params=global_p if use_fl else None,
                system=system, reward_history=reward_history)


def evaluate_variant(profiles, cfg, train_result, label):
    """Evaluate a trained ablation variant over eval_days days."""
    n_mg = cfg['system']['n_mg']
    eval_days = cfg['training']['eval_days']

    if train_result is None:
        # No PyTorch — use rule-based proxy
        rng = np.random.default_rng(cfg['training']['seed'])
        system = MultiMGSystem(profiles, n_mg=n_mg,
                               gnn_dim=cfg['gnn']['embed_dim'], rng=rng)
        return evaluate_policy(system, n_mg, eval_days, label,
                               action_fn=action_peak_shift, seed=200)

    system = train_result['system']

    if train_result['global_params'] is not None:
        return evaluate_policy(system, n_mg, eval_days, label,
                               trained_params=train_result['global_params'],
                               seed=200)
    else:
        # No FL aggregation: use the first agent's own parameters
        params = train_result['agents'][0].get_params()
        return evaluate_policy(system, n_mg, eval_days, label,
                               trained_params=params, seed=200)


def run_ablation(profiles, cfg, do_train: bool) -> pd.DataFrame:
    """Run all 4 ablation variants and return a results DataFrame."""
    variants = [
        ('DRL Only',          False, False),
        ('Fed-TD3',           True,  False),
        ('GNN-TD3',           False, True),
        ('FedGNN-TD3 (Full)', True,  True),
    ]

    rows = []
    for label, use_fl, use_gnn in variants:
        print(f"\n  --- Ablation variant: {label} "
              f"(FL={use_fl}, GNN={use_gnn}) ---")
        if do_train and TORCH:
            tr = train_variant(profiles, cfg, use_fl, use_gnn)
        else:
            tr = None
        res = evaluate_variant(profiles, cfg, tr, label)
        rows.append(dict(
            Variant=label, FL='Yes' if use_fl else 'No',
            GNN='Yes' if use_gnn else 'No',
            Cost_USD_day=round(res['cost_per_day'], 2),
            RES_Util_pct=round(res['res_util_pct'], 2),
            Violation_pct=round(res['violation_pct'], 3),
        ))
        print(f"    Cost={res['cost_per_day']:.2f} USD/day  "
              f"RES={res['res_util_pct']:.2f}%  "
              f"Viol={res['violation_pct']:.3f}%")

    return pd.DataFrame(rows)


def run_benchmarks(profiles, cfg) -> pd.DataFrame:
    """Evaluate against simple baseline heuristics (Table II style)."""
    n_mg = cfg['system']['n_mg']
    eval_days = cfg['training']['eval_days']
    rng = np.random.default_rng(cfg['training']['seed'])
    system = MultiMGSystem(profiles, n_mg=n_mg,
                           gnn_dim=cfg['gnn']['embed_dim'], rng=rng)

    print("\n  --- Benchmark comparison ---")
    methods = [
        ('Peak-Shift Heuristic', action_peak_shift),
        ('Rule-Based Greedy',    action_rule_greedy),
        ('No-Storage Reference', action_no_storage),
    ]
    rows = []
    for label, fn in methods:
        res = evaluate_policy(system, n_mg, eval_days, label,
                              action_fn=fn, seed=300)
        rows.append(dict(Method=label,
                         Cost_USD_day=round(res['cost_per_day'], 2),
                         RES_Util_pct=round(res['res_util_pct'], 2),
                         Violation_pct=round(res['violation_pct'], 3)))
        print(f"    {label}: Cost={res['cost_per_day']:.2f}  "
              f"RES={res['res_util_pct']:.2f}%  Viol={res['violation_pct']:.3f}%")

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir:
        cfg['output']['tables_dir'] = args.output_dir

    print("=" * 68)
    print("  FedGNN-TD3 — Ablation Study & Benchmark Comparison")
    print("=" * 68)
    print(f"  Train mode  : {'full TD3' if (args.train and TORCH) else 'rule-based fallback'}")

    profiles = preprocess_all(cfg)

    ablation_df  = run_ablation(profiles, cfg, do_train=args.train)
    benchmark_df = run_benchmarks(profiles, cfg)

    out_dir = pathlib.Path(cfg['output']['tables_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    ablation_df.to_csv(out_dir / 'table3_ablation.csv', index=False)
    benchmark_df.to_csv(out_dir / 'table2_benchmarks.csv', index=False)

    print(f"\n  Saved: {out_dir/'table3_ablation.csv'}")
    print(f"  Saved: {out_dir/'table2_benchmarks.csv'}")
    print("\nAblation results:\n", ablation_df.to_string(index=False))
    print("\nBenchmark results:\n", benchmark_df.to_string(index=False))


if __name__ == '__main__':
    main()
