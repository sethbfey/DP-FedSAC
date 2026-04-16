# src/scripts/train_baseline.py

# $ python scr/scripts/train_baselines.py
#     --baseline     [STR, default="fedavg"]
#     --config       [STR, default="femnist"]
#     --es_patience  [INT, default=200]
#     --em_min_delta [FLOAT, default=1e-4]
#     --sigma_t      [FLOAT, default=2.0]

import re
import sys
import yaml
import copy
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.cnn import FEMNIST_CNN
from utils.rdp import rdp_per_round

def load_config(config_path: Path):
    with config_path.open('r') as f:
        return yaml.safe_load(f)
    
def write_max_clipping_norm_to_config(config_path: Path, p75):
    content = config_path.read_text()
    content = re.sub(r'(max_clipping_norm:\s*)[0-9.]+', rf'\g<1>{p75:.4f}', content)
    config_path.write_text(content)

def get_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    elif torch.cuda.is_available(): return torch.device('cuda')
    return torch.device('cpu')

def load_data(config, dataset_name):
    clients_dir = ROOT / 'data' / 'clients' / dataset_name
    N = config['federated_learning']['num_clients']
    
    print(f"Loading {N} client datasets from {clients_dir}...")
    datasets = [
        torch.load(clients_dir / f'client_{i}.pt', weights_only=False)
        for i in range(N)
    ]
    val = torch.load(clients_dir / 'global_val.pt', weights_only=False)
    val_loader = DataLoader(val, batch_size=256, shuffle=False)
    print("Done loading\n")
    return datasets, val_loader

def select_clients(round_t, num_clients, k):
    selected_clients = np.random.default_rng(2026 + round_t).choice(num_clients, size=k, replace=False)
    return selected_clients

# Run local SGD,
# and return raw weights
def local_train(global_model, dataset, config, criterion, device):
    local_model = copy.deepcopy(global_model)
    local_model.train()

    optimizer = optim.SGD(
        local_model.parameters(),
        lr=config['federated_learning']['learning_rate']
    )

    loader = DataLoader(
        dataset,
        batch_size=config['federated_learning']['batch_size'],
        shuffle=True
    )

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

# Returns mean CE loss and accuracy on the global validation set
def eval_model(model, val_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y    = x.to(device), y.to(device)
            logits  = model(x)
            total_loss += criterion(logits, y).item() * x.size(0)
            correct    += (logits.argmax(dim=1) == y).sum().item()
            total      += x.size(0)
    model.train()
    return total_loss / total, correct / total

# Apply aggregate delta to global model parameters
def apply_aggregate(model, aggregate):
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.add_(aggregate[offset:offset + numel].view_as(p.data))
        offset += numel

# Run FedAvg, the upper bound baseline
def run_fedavg(config, config_path, client_datasets, val_loader, 
               dataset_name, es_patience=200, es_min_delta=1e-4):
    
    device    = get_device()
    criterion = nn.CrossEntropyLoss()
    model     = FEMNIST_CNN().to(device)

    K         = config['federated_learning']['clients_per_round']
    N         = config['federated_learning']['num_clients']
    T         = config['federated_learning']['num_global_steps']
    server_lr = config['federated_learning']['server_lr']
    server_mo = config['federated_learning']['server_momentum']

    all_norms = []  # All K * t raw norms up to early stop; p75 across this becomes C_max

    best_val_loss  = float('inf')
    patience_count = 0
    momentum_buf   = None

    run = wandb.init(
        project=config['wandb']['project_name'],
        entity=config['wandb']['entity'],
        name=f'FedAvg-{dataset_name.upper()}',
        group=f'{dataset_name.upper()}-baselines',
        tags=['fedavg', dataset_name, 'upper-bound'],
        config={
            'baseline':          'FedAvg',
            'dataset':           dataset_name,
            'num_clients':       N,
            'clients_per_round': K,
            'num_global_steps':  T,
            'local_epochs':      config['federated_learning']['local_epochs'],
            'learning_rate':     config['federated_learning']['learning_rate'],
            'server_lr':         server_lr,
            'batch_size':        config['federated_learning']['batch_size'],
            'server_momentum':   server_mo,
            'dp':                False,
            'client_seed':       '2026 + round',
        }
    )
    wandb.define_metric('round')
    wandb.define_metric('*', step_metric='round')

    loss, acc = eval_model(model, val_loader, criterion, device)
    print(f"[FedAvg] round=0 loss={loss:.4f} acc={acc:.4f}")

    for t in range(T):
        selected   = select_clients(t, N, K)
        raw_deltas = [
            local_train(model, client_datasets[int(cid)], config, criterion, device)
            for cid in selected
        ]

        raw_norms_t = [dw.norm(2).item() for dw in raw_deltas]
        all_norms.extend(raw_norms_t)

        with torch.no_grad():
            agg = torch.stack(raw_deltas).mean(dim=0)
            if momentum_buf is None:
                momentum_buf = agg.clone()
            else:
                momentum_buf.mul_(server_mo).add_(agg)
        apply_aggregate(model, server_lr * momentum_buf)

        loss, acc = eval_model(model, val_loader, criterion, device)

        wandb.log({
            'round':            t + 1,
            'val_loss':         loss,
            'val_accuracy':     acc,
            'grad_norm_mean':   float(np.mean(raw_norms_t)),
            'grad_norm_median': float(np.median(raw_norms_t)),
            'grad_norm_p75':    float(np.percentile(raw_norms_t, 75)),
            'grad_norm_max':    float(np.max(raw_norms_t)),
        })

        if (t + 1) % 50 == 0:
            print(f"[FedAvg] round={t+1:>4} loss={loss:.4f} acc={acc:.4f}  "
                  f"norm_p75={np.percentile(raw_norms_t, 75):.4f}  "
                  f"patience={patience_count}/{es_patience}")

        if loss < best_val_loss - es_min_delta:
            best_val_loss  = loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= es_patience:
                print(f"\n[FedAvg] Early stop at round {t+1}")
                break

    # C_max = p75
    norms_array = np.array(all_norms, dtype=np.float32)
    p75 = float(np.percentile(norms_array, 75))

    print(f"\n[FedAvg] Gradient norm statistics (all {T} rounds x {K} clients):")
    for pct in [50, 75, 90, 95]:
        print(f"         p{pct}: {np.percentile(norms_array, pct):.4f}")
    print(f"\n  -> C_max = p75 = {p75:.4f}; writing to {config_path}")

    wandb.summary['c_max_p75']      = p75
    wandb.summary['grad_norm_p50']  = float(np.percentile(norms_array, 50))
    wandb.summary['grad_norm_p90']  = float(np.percentile(norms_array, 90))
    wandb.summary['final_val_loss'] = loss
    wandb.summary['final_val_acc']  = acc
    wandb.summary['total_rounds']   = T

    wandb.finish()

    write_max_clipping_norm_to_config(config_path, p75)
    print(f"  {config_path.name} max_clipping_norm updated to {p75:.4f}\n")

    return p75

# Run DP-FedAvg, the lower bound baseline
# McMahan et al. (2018) DP-FedAvg
def run_dp_fedavg(config, client_datasets, val_loader, sigma_t, dataset_name):
    device    = get_device()
    criterion = nn.CrossEntropyLoss()
    model     = FEMNIST_CNN().to(device)

    K         = config['federated_learning']['clients_per_round']
    N         = config['federated_learning']['num_clients']
    T         = config['federated_learning']['num_global_steps']
    c_fixed   = config['differential_privacy']['max_clipping_norm']
    server_lr = config['federated_learning']['server_lr']
    server_mo = config['federated_learning']['server_momentum']

    rdp_alpha   = config['differential_privacy']['rdp_alpha']
    max_epsilon = config['differential_privacy']['max_epsilon']
    q           = K / N
    delta_eps   = rdp_per_round(int(rdp_alpha), sigma_t, q)

    accumulated_eps = 0.0
    clip_fractions  = []
    momentum_buf    = None

    run = wandb.init(
        project=config['wandb']['project_name'],
        entity=config['wandb']['entity'],
        name=f'DP-FedAvg-{dataset_name.upper()}',
        group=f'{dataset_name.upper()}-baselines',
        tags=['dp_fedavg', dataset_name, 'lower-bound'],
        config={
            'baseline':            'DP-FedAvg',
            'dataset':             dataset_name,
            'num_clients':         N,
            'clients_per_round':   K,
            'num_rounds':          T,
            'local_epochs':        config['federated_learning']['local_epochs'],
            'lr':                  config['federated_learning']['learning_rate'],
            'server_lr':           server_lr,
            'server_momentum':     server_mo,
            'batch_size':          config['federated_learning']['batch_size'],
            'c_fixed':             c_fixed,
            'sigma_t':             sigma_t,
            'rdp_alpha':           rdp_alpha,
            'max_epsilon':         max_epsilon,
            'delta_eps_per_round': delta_eps,
            'dp':                  True,
            'client_seed':         '2026 + round',
        }
    )
    wandb.define_metric('round')
    wandb.define_metric('*', step_metric='round')

    loss, acc = eval_model(model, val_loader, criterion, device)
    print(f"[DP-FedAvg sigma={sigma_t}] round=0 loss={loss:.4f} acc={acc:.4f}  "
          f"Delta epsilon/round={delta_eps:.4f} budget={max_epsilon}  "
          f"-> exhausted at round {int(max_epsilon / delta_eps)}")

    final_round = T
    for t in range(T):
        selected   = select_clients(t, N, K)
        raw_deltas = [
            local_train(model, client_datasets[int(cid)], config, criterion, device)
            for cid in selected
        ]

        raw_norms = torch.stack([dw.norm(2) for dw in raw_deltas])
        f_clip    = float((raw_norms > c_fixed).float().mean().item())
        m_norm    = float(raw_norms.median().item())
        clip_fractions.append(f_clip)

        with torch.no_grad():
            clipped = [
                dw * min(1.0, c_fixed / (dw.norm(2).item() + 1e-8))
                for dw in raw_deltas
            ]
            clipped_norms = [dw.norm(2).item() for dw in clipped]

            # DP aggregate
            agg = torch.stack(clipped).sum(dim=0)
            agg.add_(torch.randn_like(agg) * sigma_t * c_fixed)
            agg.div_(K)

            if momentum_buf is None:
                momentum_buf = agg.clone()
            else:
                momentum_buf.mul_(server_mo).add_(agg)

        apply_aggregate(model, server_lr * momentum_buf)
        accumulated_eps += delta_eps
        loss, acc = eval_model(model, val_loader, criterion, device)

        wandb.log({
            'round':                  t + 1,
            'val_loss':               loss,
            'val_accuracy':           acc,
            'accumulated_epsilon':    accumulated_eps,
            'delta_epsilon':          delta_eps,
            'f_clip':                 f_clip,
            'm_norm':                 m_norm,
            'grad_norm_mean_raw':     float(raw_norms.mean().item()),
            'grad_norm_mean_clipped': float(np.mean(clipped_norms)),
        })

        print(f"[DP-FedAvg sigma={sigma_t}] round={t+1:>4}  "
              f"loss={loss:.4f} acc={acc:.4f}  "
              f"epsilon={accumulated_eps:.3f}/{max_epsilon}  "
              f"f_clip={f_clip:.3f} m_norm={m_norm:.4f}")

        if accumulated_eps >= max_epsilon:
            final_round = t + 1
            print(f"\n[DP-FedAvg] Budget exhausted at round {final_round}.")
            break

    mean_f_clip = float(np.mean(clip_fractions))

    wandb.summary['final_val_loss']   = loss
    wandb.summary['final_val_acc']    = acc
    wandb.summary['total_rounds']     = final_round
    wandb.summary['final_epsilon']    = accumulated_eps
    wandb.summary['budget_exhausted'] = accumulated_eps >= max_epsilon
    wandb.summary['mean_f_clip']      = mean_f_clip

    wandb.finish()

    print(f"\n  mean f_clip = {mean_f_clip:.4f} — use as --target_clip_frac for adaptive run\n")
    return mean_f_clip

def parse_args():
    parser = argparse.ArgumentParser(description='Train baselines')
    
    parser.add_argument('--baseline', type=str, default='fedavg', choices=['fedavg', 'dp_fedavg'])
    parser.add_argument('--config', type=str, default='femnist')
    parser.add_argument('--es_patience', type=int, default=200)
    parser.add_argument('--es_min_delta', type=float, default=1e-4)
    parser.add_argument('--sigma_t', type=float, default=2.0)
    
    return parser.parse_args()

def main():
    args = parse_args()
    config_path = ROOT / 'src' / 'configs' / f"{args.config}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"No YAML config found at: {config_path}")

    config = load_config(config_path)
    client_datasets, val_loader = load_data(config, args.config)
    print(f"\nStarting Baseline: {args.baseline.upper()}")
    
    # FedAvg
    if args.baseline == 'fedavg':
        run_fedavg(
            config=config, 
            config_path=config_path, 
            client_datasets=client_datasets, 
            val_loader=val_loader, 
            dataset_name=args.config,
            es_patience=args.es_patience, 
            es_min_delta=args.es_min_delta
        )
    
    # DP-FedAvg
    elif args.baseline == 'dp_fedavg':
        run_dp_fedavg(
            config=config,
            client_datasets=client_datasets,
            val_loader=val_loader,
            dataset_name=args.config,
            sigma_t=args.sigma_t
        )
    
    # Adpative Clipping
    # elif args.baseline == 'adaptive':
    #     run_adaptive_clipping(...)
        
    else:
        raise ValueError(f"Unknown baseline selected: {args.baseline}")


if __name__ == '__main__':
    main()