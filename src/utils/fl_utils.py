# src/utils/fl_utils.py

import copy
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

def load_config(config_path: Path):
    with config_path.open('r') as f:
        return yaml.safe_load(f)

def get_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    elif torch.cuda.is_available(): return torch.device('cuda')
    return torch.device('cpu')

def load_data(config, dataset_name):
    clients_dir = ROOT / 'src' / 'data' / 'clients' / dataset_name
    N = config['federated_learning']['num_clients']
    print(f"Loading {N} client datasets from {clients_dir}...")
    datasets = [
        torch.load(clients_dir / f'client_{i}.pt', weights_only=False)
        for i in range(N)
    ]
    val = torch.load(clients_dir / 'global_val.pt', weights_only=False)
    print("Done.\n")
    return datasets, DataLoader(val, batch_size=256, shuffle=False)

def select_clients(round_t, num_clients, k):
    return np.random.default_rng(2026 + round_t).choice(num_clients, size=k, replace=False)

def get_dp_lr(config):
    fl = config['federated_learning']
    return fl.get('dp_learning_rate') or fl['learning_rate']

def get_dp_server_lr(config):
    fl = config['federated_learning']
    return fl.get('dp_server_lr') or fl['server_lr']

def local_train(global_model, dataset, config, criterion, device, lr=None):
    local_model = copy.deepcopy(global_model)
    local_model.train()
    if lr is None:
        lr = config['federated_learning']['learning_rate']
    optimizer = optim.SGD(local_model.parameters(), lr=lr)
    loader    = DataLoader(dataset, batch_size=config['federated_learning']['batch_size'], shuffle=True)

    for _ in range(config['federated_learning']['local_epochs']):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(local_model(x), y).backward()
            optimizer.step()

    with torch.no_grad():
        delta_w = torch.cat([
            (lp.data - gp.data).view(-1)
            for lp, gp in zip(local_model.parameters(), global_model.parameters())
        ])
    return delta_w

def eval_model(model, val_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y     = x.to(device), y.to(device)
            logits   = model(x)
            total_loss += criterion(logits, y).item() * x.size(0)
            correct    += (logits.argmax(dim=1) == y).sum().item()
            total      += x.size(0)
    model.train()
    return total_loss / total, correct / total

def apply_aggregate(model, aggregate):
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.add_(aggregate[offset:offset + numel].view_as(p.data))
        offset += numel
