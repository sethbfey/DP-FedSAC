# src/scripts/learning_rate_search.py

# $ python src/scripts/learning_rate_search.py 
#     --rounds   [INT, default=500] 
#     --patience [INT, default=200]
#     --config   [STR, default="femnist"]

import re
import sys
import copy
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import wandb

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.registry import get_model

# Client-Server LR grid
ETA_C_GRID = [0.01, 0.032, 0.1, 0.32]
ETA_S_GRID = [0.316, 1.0, 3.16, 10.0]

def load_config(path: Path):
    with path.open('r') as f:
        return yaml.safe_load(f)

def write_lr_to_config(config_path: Path, eta_c, eta_s):
    content = config_path.read_text()
    content = re.sub(r'(learning_rate:\s*)[0-9.e+-]+', rf'\g<1>{eta_c}', content)
    content = re.sub(r'(server_lr:\s*)[0-9.e+-]+', rf'\g<1>{eta_s}', content)
    config_path.write_text(content)

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
    val_loader = DataLoader(val, batch_size=256, shuffle=False)
    print("Done loading\n")
    return datasets, val_loader

# A single FedAvg run using an LR combination
def run_one(config, datasets, val_loader, eta_c, eta_s, max_rounds, patience, device, dataset_name):
    criterion = nn.CrossEntropyLoss()
    model     = get_model(dataset_name)().to(device)

    K    = config['federated_learning']['clients_per_round']
    N    = config['federated_learning']['num_clients']
    E    = config['federated_learning']['local_epochs']
    bs   = config['federated_learning']['batch_size']
    beta = config['federated_learning']['server_momentum']

    momentum_buf  = None
    best_acc      = 0.0
    patience_left = patience

    run = wandb.init(
        project=config['wandb']['project_name'],
        entity=config['wandb']['entity'],
        group=f'{dataset_name}/lr-search',
        name=f'eta_c={eta_c}/eta_s={eta_s}',
        config={
            'eta_c': eta_c, 'eta_s': eta_s, 'server_momentum': beta,
            'K': K, 'N': N, 'E': E, 'batch_size': bs,
            'max_rounds': max_rounds, 'patience': patience,
        },
        reinit=True,
    )
    wandb.define_metric('round')
    wandb.define_metric('*', step_metric='round')

    for t in range(max_rounds):
        selected = np.random.default_rng(2026 + t).choice(N, size=K, replace=False)

        raw_deltas = []

        for cid in selected:
            local = copy.deepcopy(model)
            local.train()
            opt = optim.SGD(local.parameters(), lr=eta_c)
            loader = DataLoader(datasets[int(cid)], batch_size=bs, shuffle=True)

            for _ in range(E):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    criterion(local(x), y).backward()
                    opt.step()

            with torch.no_grad():
                dw = torch.cat([
                    (lp.data - gp.data).view(-1)
                    for lp, gp in zip(local.parameters(), model.parameters())
                ])

            raw_deltas.append(dw)

        with torch.no_grad():
            agg = torch.stack(raw_deltas).mean(dim=0)

            if momentum_buf is None: momentum_buf = agg.clone()
            else: momentum_buf.mul_(beta).add_(agg)
            offset = 0
            for p in model.parameters():
                numel = p.numel()
                p.data.add_(eta_s * momentum_buf[offset:offset + numel].view_as(p.data))
                offset += numel

        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total   += x.size(0)

        model.train()
        acc = correct / total
        wandb.log({'round': t + 1, 'val_acc': acc})

        if acc > best_acc:
            best_acc      = acc
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  [ec={eta_c} es={eta_s}] early stop round {t+1}  "
                      f"peak_acc={best_acc:.4f}")
                break

        if (t + 1) % 50 == 0:
            print(
                f"  [ec={eta_c} es={eta_s}] round={t+1}  acc={acc:.4f}"
                f"  best={best_acc:.4f}  patience={patience_left}"
            )

    wandb.summary['peak_val_acc'] = best_acc
    wandb.finish()
    return best_acc

# Grid search
def main():
    parser = argparse.ArgumentParser(description='LR Search: eta_c x eta_s grid search.')
    parser.add_argument('--rounds', type=int, default=500)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--config', type=str, default='femnist')
    args = parser.parse_args()

    config_path = ROOT / 'src' / 'configs' / f"{args.config}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"No YAML config found at: {config_path}")

    config = load_config(config_path)
    device = get_device()
    datasets, val_loader = load_data(config, args.config)

    results = {}  # (eta_c, eta_s) -> peak_acc
    total = len(ETA_C_GRID) * len(ETA_S_GRID)
    done  = 0
    
    for eta_c in ETA_C_GRID:
        for eta_s in ETA_S_GRID:
            done += 1
            print(f"\n[{done}/{total}] eta_c={eta_c} eta_s={eta_s}")
            acc = run_one(config, datasets, val_loader, eta_c, eta_s, args.rounds, args.patience, device, args.config)
            results[(eta_c, eta_s)] = acc
            print(f"  -> peak_acc={acc:.4f}")

    # Select best combination
    best_pair = max(results, key=results.get)
    best_acc  = results[best_pair]
    best_ec, best_es = best_pair

    print(f"LR Search results for {args.config}:")
    for (ec, es), acc in sorted(results.items()):
        marker = " <-" if (ec, es) == best_pair else ""
        print(f"  eta_c={ec:<6} eta_s={es:<6} acc={acc:.4f}{marker}")
        
    print(f"\nBest: eta_c={best_ec} eta_s={best_es} acc={best_acc:.4f}")
    print(f"Writing to {config_path}...")

    write_lr_to_config(config_path, best_ec, best_es)
    
    print("Done")


if __name__ == '__main__':
    main()