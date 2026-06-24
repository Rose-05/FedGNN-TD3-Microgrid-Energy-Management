"""
train_federated.py
───────────────────
Full FedGNN-TD3 training loop implementing Algorithm 1 from the paper.

Algorithm 1: FedGNN-TD3 Training Algorithm for Distributed Energy Management
  1.  Initialise local theta_i and global theta_global
  2.  Construct microgrid graph G from tie-line connections
  3.  for each federated round r = 1 to R:
  4.    for each microgrid i = 1 to N in parallel:
  5.      theta_i <- theta_global                  (download global model)
  6.      for each local episode k = 1 to K:
  7.        Run local TD3 episode with privacy-preserving GNN (Eqs. 8-9)
  8.        Update theta_i using stored transitions  (local training)
  9.      Send local update delta_theta_i = theta_i - theta_global to server
  10.   Server aggregates: theta_global <- FedAvg(theta_1, ..., theta_N)
  11.   Broadcast theta_global to all microgrids
  12. return pi_global
"""

import numpy as np

try:
    import torch
    from models.td3_agent import TD3Agent
    from models.fedavg import federated_average
    TORCH = True
except ImportError:
    TORCH = False

from models.environment import MultiMGSystem


def run_fedgnn_td3(profiles: dict, cfg: dict) -> dict:
    """
    Run the full FedGNN-TD3 Algorithm 1 training loop.

    Parameters
    ----------
    profiles : dict — integrated dataset profiles (from data_mapping.build_profiles)
    cfg      : config dict (parsed config.yaml)

    Returns
    -------
    dict with training history (rounds, global_reward, mean_cost,
    mean_res_util, mean_violation, mg_losses, global_params, agents)
    or None if PyTorch is unavailable.
    """
    if not TORCH:
        print("  [SKIP] PyTorch not available — cannot run federated TD3 training.")
        print("  Install with: pip install torch")
        return None

    n_mg          = cfg['system']['n_mg']
    fed_rounds    = cfg['federated']['rounds']
    local_eps     = cfg['federated']['local_episodes']
    seed          = cfg['training']['seed']
    expl_noise    = cfg['td3']['expl_noise']

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Step 2: Construct system (graph + N microgrid environments)
    system = MultiMGSystem(profiles, n_mg=n_mg,
                           gnn_dim=cfg['gnn']['embed_dim'], rng=rng)

    sd = system.envs[0].s_dim
    ad = system.envs[0].a_dim

    # Step 1: Initialise local agents and global model
    agents   = [TD3Agent(sd, ad, cfg=cfg['td3']) for _ in range(n_mg)]
    global_p = agents[0].get_params()

    hist = dict(rounds=[], global_reward=[], mean_cost=[],
                mean_res_util=[], mean_violation=[],
                mg_losses={i: [] for i in range(n_mg)})

    print(f"  Training FedGNN-TD3: {fed_rounds} rounds x {local_eps} "
          f"local episodes per MG ...")

    # Step 3: Federated rounds
    for r in range(fed_rounds):
        local_ps, n_samples = [], []
        round_rewards, round_costs, round_utils, round_viols = [], [], [], []

        # Step 4: Local training at each MG (sequential here; can be parallelised)
        for i, (agent, env) in enumerate(zip(agents, system.envs)):
            # Step 5: Download global model
            agent.set_params(global_p)

            ep_rewards, ep_losses, ep_costs, ep_utils, ep_viols = [], [], [], [], []

            # Step 6: Local episodes
            for _ in range(local_eps):
                s = env.reset()
                ep_reward = 0.0
                ep_info = []

                # Step 7: Run TD3 episode with privacy-preserving GNN
                while True:
                    feats = system.local_features()
                    emb   = system.gnn_embed(feats)[i]
                    s[:len(emb)] = emb

                    a = agent.select_action(s, noise=expl_noise)
                    s_, reward, done, info = env.step(a, emb)
                    agent.replay_buffer.add(s, a, reward, s_, float(done))

                    s = s_
                    ep_reward += reward
                    ep_info.append(info)

                    # Step 8: Local TD3 update
                    losses = agent.train_step()
                    if losses:
                        ep_losses.append(losses.get('critic_loss', 0))

                    if done:
                        break

                ep_rewards.append(ep_reward)
                ep_costs.append(np.mean([x['cost'] for x in ep_info]))
                ep_utils.append(np.mean([x['res_util'] for x in ep_info]))
                ep_viols.append(np.mean([x['violation'] for x in ep_info]))

            hist['mg_losses'][i].append(
                float(np.mean(ep_losses)) if ep_losses else 0.0)
            round_rewards.extend(ep_rewards)
            round_costs.extend(ep_costs)
            round_utils.extend(ep_utils)
            round_viols.extend(ep_viols)

            # Step 9: Collect local update
            local_ps.append(agent.get_params())
            n_samples.append(len(agent.replay_buffer))

        # Step 10: FedAvg aggregation
        global_p = federated_average(local_ps, n_samples)
        # Step 11: Broadcast happens implicitly at the start of next round

        hist['rounds'].append(r + 1)
        hist['global_reward'].append(float(np.mean(round_rewards)))
        hist['mean_cost'].append(float(np.mean(round_costs)))
        hist['mean_res_util'].append(float(np.mean(round_utils)))
        hist['mean_violation'].append(float(np.mean(round_viols)))

        if (r + 1) % 10 == 0 or r == fed_rounds - 1:
            print(f"    Round {r+1:3d}/{fed_rounds} | "
                  f"Reward={hist['global_reward'][-1]:8.3f} | "
                  f"RES={hist['mean_res_util'][-1]*100:.1f}% | "
                  f"Viol={hist['mean_violation'][-1]:.3f}%")

    # Step 12: Return global policy (+ extra info for evaluation/plotting)
    hist['global_params'] = global_p
    hist['agents']        = agents
    hist['system']        = system
    return hist


if __name__ == '__main__':
    import yaml
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)
    from scripts.preprocess_ausgrid import preprocess_all
    profiles = preprocess_all(cfg)
    hist = run_fedgnn_td3(profiles, cfg)
    if hist:
        print(f"\nFinal global reward: {hist['global_reward'][-1]:.3f}")
