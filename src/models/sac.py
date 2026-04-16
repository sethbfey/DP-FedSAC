# src/models/sac.py

import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((
            np.array(obs,      dtype=np.float32),
            np.array(action,   dtype=np.float32),
            float(reward),
            np.array(next_obs, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, acts, rews, next_obs, dones = zip(*batch)
        return (
            torch.tensor(np.stack(obs),      dtype=torch.float32),
            torch.tensor(np.stack(acts),     dtype=torch.float32),
            torch.tensor(np.array(rews),     dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.stack(next_obs), dtype=torch.float32),
            torch.tensor(np.array(dones),    dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def _distribution(self, obs):
        h       = self.net(obs)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std.exp()

    def sample(self, obs):
        mean, std = self._distribution(obs)
        eps       = torch.randn_like(mean)
        x_t       = mean + std * eps
        action    = torch.tanh(x_t)
        log_prob  = (
            torch.distributions.Normal(mean, std).log_prob(x_t)
            - torch.log(1.0 - action.pow(2) + 1e-6)
        ).sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, obs):
        mean, _ = self._distribution(obs)
        return torch.tanh(mean)


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))

class SAC:
    """
    SAC agent for DP-FedSAC.

    Observations:
        dim 0  epsilon_t    -> epsilon_t / max_epsilon
        dim 1  t_norm       -> already in [0, 1]
        dim 2  f_clip       -> already in [0, 1]
        dim 3  m_norm       -> m_norm / (2 * max_clipping_norm)
        dim 4  C_prev       -> C_prev / max_clipping_norm
        dim 5  sigma_prev   -> (sigma_prev - min_sigma) / (max_sigma - min_sigma)

    Raw observations are stored in the replay buffer; normalisation is applied
    at sample time in update() and at inference time in select_action().
    """

    def __init__(self, obs_dim, action_dim, config):
        sac_cfg = config['sac_agent']
        dp_cfg  = config['differential_privacy']

        self.gamma      = sac_cfg['gamma']
        self.tau        = sac_cfg['tau_rho']
        self.batch_size = sac_cfg['batch_size']

        self._obs_shift = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0,
            dp_cfg['min_noise_multiplier'],
        ], dtype=np.float32)

        self._obs_scale = np.array([
            dp_cfg['max_epsilon'],
            1.0,
            1.0,
            2.0 * dp_cfg['max_clipping_norm'],
            dp_cfg['max_clipping_norm'],
            dp_cfg['max_noise_multiplier'] - dp_cfg['min_noise_multiplier'],
        ], dtype=np.float32)

        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Networks
        self.actor   = Actor(obs_dim, action_dim).to(self.device)
        self.critic1 = Critic(obs_dim, action_dim).to(self.device)
        self.critic2 = Critic(obs_dim, action_dim).to(self.device)
        self.target1 = Critic(obs_dim, action_dim).to(self.device)
        self.target2 = Critic(obs_dim, action_dim).to(self.device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_opt   = optim.Adam(self.actor.parameters(),   lr=sac_cfg['actor_lr'])
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=sac_cfg['critic_lr'])
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=sac_cfg['critic_lr'])

        # Auto-tuned entropy temperature
        self.target_entropy = -float(action_dim)
        self.log_alpha      = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt      = optim.Adam([self.log_alpha], lr=sac_cfg['actor_lr'])

        self.replay_buffer = ReplayBuffer(sac_cfg['buffer_size'])

    def _norm_np(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self._obs_shift) / (self._obs_scale + 1e-8)

    def _norm_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        shift = torch.as_tensor(self._obs_shift, device=self.device)
        scale = torch.as_tensor(self._obs_scale, device=self.device)
        return (obs - shift) / (scale + 1e-8)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_n = self._norm_np(obs)
        t     = torch.tensor(obs_n, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor.deterministic(t) if deterministic else self.actor.sample(t)[0]
        return action.cpu().numpy()[0]

    def update(self) -> dict:
        if len(self.replay_buffer) < self.batch_size:
            return {}

        obs, actions, rewards, next_obs, dones = [
            x.to(self.device) for x in self.replay_buffer.sample(self.batch_size)
        ]
        obs_n      = self._norm_tensor(obs)
        next_obs_n = self._norm_tensor(next_obs)
        alpha      = self.log_alpha.exp().detach()

        # Critic targets
        with torch.no_grad():
            next_a, next_log_pi = self.actor.sample(next_obs_n)
            q_next   = torch.min(self.target1(next_obs_n, next_a),
                                 self.target2(next_obs_n, next_a))
            target_q = rewards + self.gamma * (1.0 - dones) * (q_next - alpha * next_log_pi)

        # Critic losses
        q1_loss = F.mse_loss(self.critic1(obs_n, actions), target_q)
        q2_loss = F.mse_loss(self.critic2(obs_n, actions), target_q)

        self.critic1_opt.zero_grad(); q1_loss.backward(); self.critic1_opt.step()
        self.critic2_opt.zero_grad(); q2_loss.backward(); self.critic2_opt.step()

        # Actor loss
        a_pi, log_pi = self.actor.sample(obs_n)
        q_pi         = torch.min(self.critic1(obs_n, a_pi), self.critic2(obs_n, a_pi))
        actor_loss   = (alpha * log_pi - q_pi).mean()

        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        # Temperature loss
        alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        # Update target networks with EMA
        for p, tp in zip(self.critic1.parameters(), self.target1.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        for p, tp in zip(self.critic2.parameters(), self.target2.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        return {
            'critic1_loss': q1_loss.item(),
            'critic2_loss': q2_loss.item(),
            'actor_loss':   actor_loss.item(),
            'alpha':        self.log_alpha.exp().item(),
        }

    # Checkpointing
    def save(self, path: str, episode: int = 0, total_steps: int = 0, wandb_run_id: str = None):
        torch.save({
            # networks
            'actor':        self.actor.state_dict(),
            'critic1':      self.critic1.state_dict(),
            'critic2':      self.critic2.state_dict(),
            'target1':      self.target1.state_dict(),
            'target2':      self.target2.state_dict(),
            'actor_opt':    self.actor_opt.state_dict(),
            'critic1_opt':  self.critic1_opt.state_dict(),
            'critic2_opt':  self.critic2_opt.state_dict(),
            'alpha_opt':    self.alpha_opt.state_dict(),
            'log_alpha':    self.log_alpha.item(),
            'replay_buffer': list(self.replay_buffer.buffer),
            'episode':      episode,
            'total_steps':  total_steps,
            'wandb_run_id': wandb_run_id,
        }, path)

    # Resume SAC agent from checkpoint
    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.actor.load_state_dict(ckpt['actor'])
        self.critic1.load_state_dict(ckpt['critic1'])
        self.critic2.load_state_dict(ckpt['critic2'])
        self.target1.load_state_dict(ckpt['target1'])
        self.target2.load_state_dict(ckpt['target2'])

        self.actor_opt.load_state_dict(ckpt['actor_opt'])
        self.critic1_opt.load_state_dict(ckpt['critic1_opt'])
        self.critic2_opt.load_state_dict(ckpt['critic2_opt'])
        self.alpha_opt.load_state_dict(ckpt['alpha_opt'])

        with torch.no_grad():
            self.log_alpha.fill_(ckpt['log_alpha'])

        self.replay_buffer.buffer.clear()
        self.replay_buffer.buffer.extend(ckpt['replay_buffer'])

        return ckpt['episode'], ckpt['total_steps']
