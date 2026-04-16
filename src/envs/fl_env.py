# src/envs/fl_env.py

import copy
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.cnn import FEMNIST_CNN
from utils.rdp import rdp_per_round

# Synchronous DP-FedAvg environment for the DP-FedSAC agent.
class FL_DP_Env(gym.Env):
    def __init__(self, config, is_training_agent=True):
        super(FL_DP_Env, self).__init__()

        self.config = config

        # FL params
        self.num_clients       = config['federated_learning']['num_clients']
        self.clients_per_round = config['federated_learning']['clients_per_round']
        self.max_rounds        = config['federated_learning']['num_global_steps']
        self.local_epochs      = config['federated_learning']['local_epochs']
        self.lr                = config['federated_learning']['learning_rate']
        self.server_lr         = config['federated_learning']['server_lr']
        self.batch_size        = config['federated_learning']['batch_size']
        self.server_momentum   = config['federated_learning']['server_momentum']

        # DP params
        self.rdp_alpha   = config['differential_privacy']['rdp_alpha']
        self.max_epsilon = config['differential_privacy']['max_epsilon']
        self.max_clip    = config['differential_privacy']['max_clipping_norm']
        self.max_sigma   = config['differential_privacy']['max_noise_multiplier']
        self.min_sigma   = config['differential_privacy']['min_noise_multiplier']

        # Reward param
        self.beta = config['sac_agent']['reward_beta']

        # if True (agent training), randomize client selection each episode
        # if False (baseline comparison), use per-round deterministic seed
        self.is_training_agent = is_training_agent

        # Action space: [a_C, a_sigma] in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: [epsilon_t, t_norm, f_clip, m_norm, C_prev, sigma_prev]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, self.min_sigma], dtype=np.float32),
            high=np.array([self.max_epsilon, 1.0, 1.0, np.inf,  self.max_clip, self.max_sigma], dtype=np.float32),
            dtype=np.float32
        )

        # Episode states (reset variables)
        self.current_round   = 0
        self.current_epsilon = 0.0
        self.C_prev          = self.max_clip
        self.sigma_prev      = self.min_sigma
        self.momentum_buf    = None 

        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Load client data and global validation data
        ROOT        = Path(__file__).resolve().parent.parent.parent
        dataset     = config['federated_learning']['dataset']
        clients_dir = ROOT / 'src' / 'data' / 'clients' / dataset

        print(f"Loading {self.num_clients} client datasets from {clients_dir}...")
        self.client_datasets = [
            torch.load(clients_dir / f'client_{i}.pt', weights_only=False)
            for i in range(self.num_clients)
        ]
        val_dataset = torch.load(clients_dir / 'global_val.pt', weights_only=False)
        self.val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        print("Datasets loaded.")

        self.criterion = nn.CrossEntropyLoss()
        self.global_model = FEMNIST_CNN().to(self.device)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.current_round   = 0
        self.current_epsilon = 0.0
        self.C_prev          = self.max_clip
        self.sigma_prev      = self.min_sigma
        self.momentum_buf    = None

        self.global_model = FEMNIST_CNN().to(self.device)
        return self._make_obs(f_clip=0.0, m_norm=0.0), {}

    def step(self, action):
        # Scale actions
        C_t     = ((action[0] + 1.0) / 2.0) * self.max_clip
        sigma_t = ((action[1] + 1.0) / 2.0) * (self.max_sigma - self.min_sigma) + self.min_sigma

        # Per-round RDP cost
        q = self.clients_per_round / self.num_clients
        delta_epsilon = rdp_per_round(int(self.rdp_alpha), sigma_t, q)
        self.current_epsilon += delta_epsilon

        # Evaluate accuracy before global model update (for reward calc)
        acc_before = self._eval_accuracy()

        # Sample K clients; deterministic per round when not training the agent
        if self.is_training_agent:
            client_rng = np.random.default_rng()
        else:
            client_rng = np.random.default_rng(2026 + self.current_round)
        selected = client_rng.choice(self.num_clients, size=self.clients_per_round, replace=False)

        # Fetch raw client updates
        raw_deltas = [self._local_train(int(cid)) for cid in selected]

        # Raw client update stats
        raw_norms = torch.stack([dw.norm(2) for dw in raw_deltas])
        f_clip    = float((raw_norms > C_t).float().mean().item())
        m_norm    = float(raw_norms.median().item())

        # DP-FedAvg: clip -> sum -> add noise -> divide by K -> momentum -> server step
        with torch.no_grad():
            clipped = [
                dw * min(1.0, C_t / (dw.norm(2).item() + 1e-8))
                for dw in raw_deltas
            ]

            # Δ̃^t = (sum(clipped) + N(0, σ²C²I)) / K
            agg = torch.stack(clipped).sum(dim=0)
            agg.add_(torch.randn_like(agg) * sigma_t * C_t)
            agg.div_(self.clients_per_round)

            # Δ̄^t = β·Δ̄^{t-1} + Δ̃^t
            if self.momentum_buf is None:
                self.momentum_buf = agg.clone()
            else:
                self.momentum_buf.mul_(self.server_momentum).add_(agg)

            # θ^{t+1} = θ^t + η_s·Δ̄^t
            offset = 0
            for p in self.global_model.parameters():
                numel = p.numel()
                p.data.add_(self.server_lr * self.momentum_buf[offset:offset + numel].view_as(p.data))
                offset += numel

        acc_after = self._eval_accuracy()

        reward = (acc_after - acc_before) - self.beta * (delta_epsilon / self.max_epsilon)

        # Advance to next round
        self.C_prev     = C_t
        self.sigma_prev = sigma_t
        self.current_round += 1

        terminated = self.current_epsilon >= self.max_epsilon
        truncated  = (not terminated) and (self.current_round >= self.max_rounds)

        obs  = self._make_obs(f_clip, m_norm)
        info = {
            'C_t':           C_t,
            'sigma_t':       sigma_t,
            'delta_epsilon': delta_epsilon,
            'f_clip':        f_clip,
            'm_norm':        m_norm,
            'acc_before':    acc_before,
            'acc_after':     acc_after,
        }

        return obs, reward, terminated, truncated, info

    def _make_obs(self, f_clip, m_norm):
        return np.array([
            float(self.current_epsilon),
            float(self.current_round / self.max_rounds),
            float(f_clip),
            float(m_norm),
            float(self.C_prev),
            float(self.sigma_prev),
        ], dtype=np.float32)

    # Local SGD
    def _local_train(self, client_id):
        local_model = copy.deepcopy(self.global_model)
        local_model.train()
        optimizer = optim.SGD(local_model.parameters(), lr=self.lr)
        loader    = DataLoader(
            self.client_datasets[client_id],
            batch_size=self.batch_size,
            shuffle=True
        )

        for _ in range(self.local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                self.criterion(local_model(x), y).backward()
                optimizer.step()

        with torch.no_grad():
            delta_w = torch.cat([
                (lp.data - gp.data).view(-1)
                for lp, gp in zip(local_model.parameters(), self.global_model.parameters())
            ])
        return delta_w

    # Global model evaulation on global validation set
    def _eval_accuracy(self):
        self.global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y    = x.to(self.device), y.to(self.device)
                correct += (self.global_model(x).argmax(dim=1) == y).sum().item()
                total   += x.size(0)
        self.global_model.train()
        return correct / total