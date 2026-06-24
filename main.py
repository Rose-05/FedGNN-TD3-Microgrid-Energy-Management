"""
main.py — FedGNN-TD3 entry point.
Loads config, runs preprocessing, training, and evaluation.

Usage
──────
    python main.py                         # evaluate with rule-based policies
    python main.py --train                 # full TD3 training + evaluation
    python main.py --train --fed_rounds 100 --local_episodes 20 --seed 7
"""

import argparse
import pathlib
import sys
import json

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import yaml
import pandas as pd

from scripts.preprocess_ausgrid import preprocess_all
from training.train_federated   import run_fedgnn_td3
from training.evaluate          import run_all_evaluations, print_results
from scripts.plot_results       import generate_all_figures


def parse_args():
    p = argparse.ArgumentParser(description='FedGNN-TD3 main entry point')
    p.add_argument('--config',         default='config.yaml')
    p.add_argument('--train',          action='store_true')
    p.add_argument('--evaluate',       action='store_true')
    p.add_argument('--seed',           type=int, default=None)
    p.add_argument('--fed_rounds',     type=int, default=None)
    p.add_argument('--local_episodes', type=int, default=None)
    p.add_argument('--eval_days',      type=int, default=None)
    p.add_argument('--output_dir',     default=None)
    p.add_argument('--no_figures',     action='store_true')
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, args):
    if args.seed is not None:           cfg['training']['seed'] = args.seed
    if args.fed_rounds is not None:     cfg['federated']['rounds'] = args.fed_rounds
    if args.local_episodes is not None: cfg['federated']['local_episodes'] = args.local_episodes
    if args.eval_days is not None:      cfg['training']['eval_days'] = args.eval_days
    if args.output_dir is not None:
        cfg['output']['figures_dir'] = str(pathlib.Path(args.output_dir) / 'figures')
        cfg['output']['tables_dir']  = str(pathlib.Path(args.output_dir) / 'tables')
    return cfg


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    cfg  = apply_overrides(cfg, args)

    for d in (cfg['output']['figures_dir'], cfg['output']['tables_dir'],
              cfg['data']['processed_dir']):
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  FedGNN-TD3: Distributed Renewable Energy Management")
    print("=" * 68)
    print(f"  Config      : {args.config}")
    print(f"  Seed        : {cfg['training']['seed']}")
    print(f"  Fed rounds  : {cfg['federated']['rounds']}")
    print(f"  Eval days   : {cfg['training']['eval_days']}")

    print("\n── Step 1: Preprocessing datasets ──────────────────────────────")
    profiles = preprocess_all(cfg)

    train_hist     = None
    trained_params = None
    if args.train:
        print("\n── Step 2: Training FedGNN-TD3 (Algorithm 1) ───────────────────")
        train_hist = run_fedgnn_td3(profiles, cfg)
        if train_hist:
            trained_params = train_hist.get('global_params')
            ckpt = pathlib.Path(cfg['output']['tables_dir']) / 'training_history.json'
            safe = {k: v for k, v in train_hist.items()
                   if k not in ('agents', 'global_params', 'system')}
            with open(ckpt, 'w') as f:
                json.dump(safe, f, indent=2, default=str)
            print(f"  Training history saved -> {ckpt}")
    else:
        print("\n── Step 2: Training skipped (add --train to run) ───────────────")

    print("\n── Step 3: Evaluating policies ──────────────────────────────────")
    eval_results = run_all_evaluations(profiles, cfg, trained_params)
    print_results(eval_results, cfg)

    df = pd.DataFrame([
        dict(method=r['label'],
            cost_usd_day=round(r['cost_per_day'], 4),
            res_util_pct=round(r['res_util_pct'], 4),
            violation_pct=round(r['violation_pct'], 6))
        for r in eval_results
    ])
    csv_p = pathlib.Path(cfg['output']['tables_dir']) / 'results_computed.csv'
    df.to_csv(csv_p, index=False)
    print(f"\n  Results -> {csv_p}")

    if not args.no_figures:
        print("\n── Step 4: Generating figures ───────────────────────────────────")
        generate_all_figures(profiles, eval_results, train_hist, cfg)

    print("\n" + "=" * 68)
    print(f"  Complete. Outputs in: {cfg['output']['figures_dir']}")
    print("=" * 68 + "\n")


if __name__ == '__main__':
    main()
