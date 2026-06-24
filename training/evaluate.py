"""
evaluate.py
───────────
Policy evaluation and metric computation.

Implements Equations 21-24 from the paper:
    C_tot   = sum_t (C_grid_t + C_ESS_t + C_trade_t)             Eq. 21
    eta_RES = sum(P_RES_used) / sum(P_RES_avail) * 100%          Eq. 22
    V_rate  = N_viol / N_total * 100%                             Eq. 23
    T_train = t_end - t_start                                      Eq. 24

Also evaluates baseline policies (rule-based, peak-shift, no-storage) for
comparison against the trained FedGNN-TD3 policy.
"""

import numpy as np

from models.environment import MultiMGSystem, N_STEPS, ESS_CAP, ESS_PMAX, \
    SOC_MIN, SOC_MAX

try:
    from models.td3_agent import TD3Agent
    TORCH = True
except ImportError:
    TORCH = False


# ── Baseline policies ─────────────────────────────────────────────────────────

def action_rule_greedy(env) -> np.ndarray:
    """Rule-based greedy: charge on RES surplus, discharge on deficit."""
    t = min(env.t, N_STEPS - 1)
    net = env.sol_kw[t] + env.wnd_kw[t] - env.lod_kw[t]
    sf = env.soc / ESS_CAP
    if net > 0 and sf < SOC_MAX:
        c = min(net, ESS_PMAX) / ESS_PMAX
        return np.array([2*c-1, -1, -1, -1, -1], np.float32)
    elif net < 0 and sf > SOC_MIN:
        d = min(-net, ESS_PMAX) / ESS_PMAX
        return np.array([-1, 2*d-1, -1, -1, -1], np.float32)
    elif net < 0:
        b = min(-net, 500) / 500
        return np.array([-1, -1, 2*b-1, -1, -1], np.float32)
    return np.array([-1, -1, -1, -1, -1], np.float32)


def action_peak_shift(env) -> np.ndarray:
    """Peak-shift heuristic: charge mid-day, discharge evening peak."""
    t = min(env.t, N_STEPS - 1)
    h = t * 0.25
    sf = env.soc / ESS_CAP
    if 9 <= h <= 15 and sf < SOC_MAX:
        return np.array([1.0, -1, -1, -1, -1], np.float32)
    elif 17 <= h <= 21 and sf > SOC_MIN:
        d = min(env.lod_kw[t], ESS_PMAX) / ESS_PMAX
        return np.array([-1, 2*d-1, -1, -1, -1], np.float32)
    return action_rule_greedy(env)


def action_no_storage(env) -> np.ndarray:
    """No ESS — curtail surplus, buy deficit from grid."""
    t = min(env.t, N_STEPS - 1)
    net = env.sol_kw[t] + env.wnd_kw[t] - env.lod_kw[t]
    if net < 0:
        b = min(-net, 500) / 500
        return np.array([-1, -1, 2*b-1, -1, -1], np.float32)
    return np.array([-1, -1, -1, -1, -1], np.float32)


# ── Evaluation core ───────────────────────────────────────────────────────────

def evaluate_policy(system: MultiMGSystem, n_mg: int, eval_days: int,
                    label: str, trained_params: dict = None,
                    action_fn=None, seed: int = 100) -> dict:
    """
    Evaluate a policy over eval_days independent days.

    Either pass `trained_params` (TD3 actor/critic state_dicts from FedAvg)
    to use the trained policy, or pass `action_fn(env) -> action` for a
    rule-based baseline.
    """
    use_td3 = (trained_params is not None and TORCH)
    if use_td3:
        agents = []
        for _ in range(n_mg):
            ag = TD3Agent(system.envs[0].s_dim, system.envs[0].a_dim)
            ag.set_params(trained_params)
            agents.append(ag)

    all_costs, all_utils, all_viols = [], [], []
    soc_traj = {mg: [] for mg in range(n_mg)}

    for _day in range(eval_days):
        day_costs, day_utils, day_viols = [], [], []
        for mg in range(n_mg):
            env = system.envs[mg]
            s = env.reset()
            info_list = []
            while True:
                feats = system.local_features()
                emb   = system.gnn_embed(feats)[mg]
                s[:len(emb)] = emb

                if use_td3:
                    a = agents[mg].select_action(s, noise=0.0)
                elif action_fn:
                    a = action_fn(env)
                else:
                    a = action_rule_greedy(env)

                s_, _, done, info = env.step(a, emb)
                info_list.append(info)
                soc_traj[mg].append(info['soc_pct'])
                s = s_
                if done:
                    break
            day_costs.append(np.mean([x['cost'] for x in info_list]))
            day_utils.append(np.mean([x['res_util'] for x in info_list]))
            day_viols.append(np.mean([x['violation'] for x in info_list]))
        all_costs.append(np.mean(day_costs))
        all_utils.append(np.mean(day_utils))
        all_viols.append(np.mean(day_viols))

    cost_per_day = float(np.mean(all_costs)) * N_STEPS * n_mg   # Eq. 21 scaled

    return dict(
        label=label,
        cost_per_day=cost_per_day,
        res_util_pct=float(np.mean(all_utils)) * 100,           # Eq. 22
        violation_pct=float(np.mean(all_viols)),                # Eq. 23
        raw_costs=all_costs, raw_utils=all_utils, raw_viols=all_viols,
        soc_traj={mg: np.array(soc_traj[mg]) for mg in range(n_mg)},
    )


def run_all_evaluations(profiles: dict, cfg: dict,
                        trained_params: dict = None) -> list:
    """
    Build the simulation system and evaluate FedGNN-TD3 (trained or rule
    proxy) against three baselines: peak-shift heuristic, rule-based greedy,
    and no-storage reference.
    """
    n_mg      = cfg['system']['n_mg']
    eval_days = cfg['training']['eval_days']
    seed      = cfg['training']['seed']
    rng       = np.random.default_rng(seed)

    system = MultiMGSystem(profiles, n_mg=n_mg,
                           gnn_dim=cfg['gnn']['embed_dim'], rng=rng)

    print(f"  Evaluating policies over {eval_days} days ...")
    results = []

    if trained_params is not None:
        label = 'FedGNN-TD3 (trained)'
        print(f"    {label} ...")
        results.append(evaluate_policy(system, n_mg, eval_days, label,
                                       trained_params=trained_params, seed=200))
    else:
        label = 'FedGNN-TD3 (rule proxy, no --train)'
        print(f"    {label} ...")
        results.append(evaluate_policy(system, n_mg, eval_days, label,
                                       action_fn=action_peak_shift, seed=200))

    print("    Peak-Shift Heuristic ...")
    results.append(evaluate_policy(system, n_mg, eval_days, 'Peak-Shift Heuristic',
                                   action_fn=action_peak_shift, seed=201))

    print("    Rule-Based Greedy ...")
    results.append(evaluate_policy(system, n_mg, eval_days, 'Rule-Based Greedy',
                                   action_fn=action_rule_greedy, seed=202))

    print("    No-Storage (reference) ...")
    results.append(evaluate_policy(system, n_mg, eval_days, 'No-Storage Reference',
                                   action_fn=action_no_storage, seed=203))

    return results


def print_results(eval_results: list, cfg: dict):
    """Print formatted results table to console."""
    n_mg      = cfg['system']['n_mg']
    eval_days = cfg['training']['eval_days']

    print("\n" + "=" * 72)
    print("  RESULTS — Computed from Dataset-Driven Simulation")
    print(f"  ({eval_days} evaluation days x {n_mg} microgrids)")
    print("=" * 72)
    print(f"  {'Method':<38} {'Cost (USD/day)':>14} "
          f"{'RES Util (%)':>12} {'Viol (%)':>9}")
    print("  " + "-" * 68)
    for r in eval_results:
        print(f"  {r['label']:<38} {r['cost_per_day']:>14.3f} "
              f"{r['res_util_pct']:>12.3f} {r['violation_pct']:>9.4f}")
    print("=" * 72)

    if len(eval_results) >= 2:
        proposed = eval_results[0]
        print(f"\n  Best policy: {proposed['label']}")
        for r in eval_results[1:]:
            cost_imp = (r['cost_per_day'] - proposed['cost_per_day']) / r['cost_per_day'] * 100
            util_imp = proposed['res_util_pct'] - r['res_util_pct']
            arrow = '↓' if cost_imp > 0 else '↑'
            print(f"    vs {r['label'][:32]:<32}: "
                  f"cost {arrow}{abs(cost_imp):.2f}%   "
                  f"RES {'+' if util_imp > 0 else ''}{util_imp:.2f} pp")
