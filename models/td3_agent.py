"""
td3_agent.py
────────────
Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.

Implements Equations 11-13 and Algorithm 1 steps 6-9 from the paper.

Three improvements over DDPG (Section II-C):
    1. Clipped double-Q learning   — reduces overestimation bias
    2. Delayed policy updates      — reduces variance (every d critic updates)
    3. Target policy smoothing     — reduces variance

State  s_i,t (Eq. 9):  [h_i,t (GNN embed) | P_RES | P_load | SoC | lambda | P_grid]
Action a_i,t (Eq. 10): [P_ch | P_dis | P_buy | P_sell | P_shed]
"""

import copy
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH = True
except ImportError:
    TORCH = False


DEFAULT_TD3_CFG = dict(
    actor_lr=3e-4, critic_lr=3e-4, gamma=0.99, tau=0.005,
    policy_delay=2, expl_noise=0.1, policy_noise=0.2,
    noise_clip=0.5, batch_size=256, buffer_size=int(1e6), hidden_dim=256,
)


class ReplayBuffer:
    """Experience replay buffer D for off-policy TD3 updates (Eq. 13)."""

    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(1e6)):
        self.max_size = max_size
        self.ptr = self.size = 0
        self.S  = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.A  = np.zeros((max_size, action_dim), dtype=np.float32)
        self.R  = np.zeros((max_size, 1),          dtype=np.float32)
        self.S_ = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.D  = np.zeros((max_size, 1),          dtype=np.float32)

    def add(self, s, a, r, s_, done):
        i = self.ptr
        self.S[i], self.A[i], self.R[i] = s, a, r
        self.S_[i], self.D[i] = s_, done
        self.ptr  = (i + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (torch.FloatTensor(self.S[idx]),
                torch.FloatTensor(self.A[idx]),
                torch.FloatTensor(self.R[idx]),
                torch.FloatTensor(self.S_[idx]),
                torch.FloatTensor(self.D[idx]))

    def __len__(self):
        return self.size


if TORCH:

    class Actor(nn.Module):
        """TD3 Actor network pi_theta -> deterministic action (Eq. 10)."""

        def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
            super().__init__()
            self.max_action = max_action
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, action_dim), nn.Tanh())

        def forward(self, state):
            return self.max_action * self.net(state)


    class Critic(nn.Module):
        """TD3 Twin Critic networks Q_phi1, Q_phi2 (clipped double-Q)."""

        def __init__(self, state_dim, action_dim, hidden_dim=256):
            super().__init__()
            def make_q():
                return nn.Sequential(
                    nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, 1))
            self.q1, self.q2 = make_q(), make_q()

        def forward(self, state, action):
            sa = torch.cat([state, action], dim=-1)
            return self.q1(sa), self.q2(sa)

        def q1_only(self, state, action):
            return self.q1(torch.cat([state, action], dim=-1))


    class TD3Agent:
        """
        TD3 agent implementing Algorithm 1 Steps 6-9.

        get_params() / set_params() expose actor + critic state_dicts
        for federated averaging (see models/fedavg.py).
        """

        def __init__(self, state_dim, action_dim, max_action=1.0,
                     device='cpu', cfg=None):
            cfg = cfg or DEFAULT_TD3_CFG
            self.device = torch.device(device)
            self.max_action = max_action
            self.gamma = cfg['gamma']; self.tau = cfg['tau']
            self.policy_delay = cfg['policy_delay']
            self.policy_noise = cfg['policy_noise']
            self.noise_clip   = cfg['noise_clip']
            self.batch_size   = cfg['batch_size']
            self.total_it = 0
            hidden_dim = cfg['hidden_dim']

            self.actor        = Actor(state_dim, action_dim, hidden_dim, max_action).to(self.device)
            self.actor_target = copy.deepcopy(self.actor)
            self.actor_opt    = optim.Adam(self.actor.parameters(), lr=cfg['actor_lr'])

            self.critic        = Critic(state_dim, action_dim, hidden_dim).to(self.device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_opt     = optim.Adam(self.critic.parameters(), lr=cfg['critic_lr'])

            self.replay_buffer = ReplayBuffer(state_dim, action_dim, cfg['buffer_size'])

        def select_action(self, state, noise=0.0):
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            a = self.actor(s).cpu().data.numpy().flatten()
            if noise > 0:
                a += np.random.randn(len(a)) * noise
            return np.clip(a, -self.max_action, self.max_action)

        def train_step(self):
            """One TD3 training step — Eqs. 12-13. Returns loss dict."""
            if len(self.replay_buffer) < self.batch_size:
                return {}
            self.total_it += 1
            s, a, r, s_, done = self.replay_buffer.sample(self.batch_size)
            s, a, r, s_, done = (x.to(self.device) for x in (s, a, r, s_, done))

            with torch.no_grad():
                noise = (torch.randn_like(a) * self.policy_noise
                         ).clamp(-self.noise_clip, self.noise_clip)
                a_ = (self.actor_target(s_) + noise).clamp(-self.max_action, self.max_action)
                q1_, q2_ = self.critic_target(s_, a_)
                q_target = r + self.gamma * (1 - done) * torch.min(q1_, q2_)

            q1, q2 = self.critic(s, a)
            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
            self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

            actor_loss_val = 0.0
            if self.total_it % self.policy_delay == 0:
                actor_loss = -self.critic.q1_only(s, self.actor(s)).mean()
                self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
                actor_loss_val = actor_loss.item()
                for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                    tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
                for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                    tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss_val}

        def get_params(self) -> dict:
            """Extract trainable parameters for FedAvg (Eq. 14)."""
            return {'actor':  copy.deepcopy(self.actor.state_dict()),
                    'critic': copy.deepcopy(self.critic.state_dict())}

        def set_params(self, params: dict):
            """Load global parameters from federated server (Eq. 15)."""
            self.actor.load_state_dict(params['actor'])
            self.critic.load_state_dict(params['critic'])
            self.actor_target  = copy.deepcopy(self.actor)
            self.critic_target = copy.deepcopy(self.critic)
