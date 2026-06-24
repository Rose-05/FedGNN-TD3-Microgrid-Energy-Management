"""
train_local.py
──────────────
Single-microgrid local TD3 training loop (no federation).
Used as a building block for train_federated.py and as the
"DRL Only" baseline in the ablation study (Table III).
"""

import numpy as np

try:
    import torch
    from models.td3_agent import TD3Agent
    TORCH = True
except ImportError:
    TORCH = False


def train_local_td3(env, gnn_embed_fn, n_episodes: int, cfg: dict,
                    expl_noise: float = 0.1):
    """
    Train a single TD3 agent on one microgrid environment, without
    any federated aggregation.

    Parameters
    ----------
    env           : MicrogridEnv instance
    gnn_embed_fn  : callable() -> np.ndarray (gnn_dim,) — current GNN embedding
                    (pass a zero vector if not using a GNN, e.g. "DRL Only")
    n_episodes    : number of training episodes
    cfg           : config dict (uses cfg['td3'])

    Returns
    -------
    agent  : trained TD3Agent
    history: dict with per-episode reward / cost / res_util / violation lists
    """
    if not TORCH:
        raise RuntimeError("PyTorch is required for train_local_td3(). "
                           "Install with: pip install torch")

    agent = TD3Agent(env.s_dim, env.a_dim, cfg=cfg.get('td3'))
    history = dict(reward=[], cost=[], res_util=[], violation=[])

    for ep in range(n_episodes):
        s = env.reset()
        ep_reward = 0.0
        ep_info = []

        while True:
            emb = gnn_embed_fn()
            s[:len(emb)] = emb
            a = agent.select_action(s, noise=expl_noise)
            s_, reward, done, info = env.step(a, emb)
            agent.replay_buffer.add(s, a, reward, s_, float(done))
            s = s_
            ep_reward += reward
            ep_info.append(info)
            agent.train_step()
            if done:
                break

        history['reward'].append(ep_reward)
        history['cost'].append(np.mean([x['cost'] for x in ep_info]))
        history['res_util'].append(np.mean([x['res_util'] for x in ep_info]))
        history['violation'].append(np.mean([x['violation'] for x in ep_info]))

    return agent, history


if __name__ == '__main__':
    print("This module is normally called from training/train_federated.py "
          "or scripts/run_experiments.py for the ablation study.")
