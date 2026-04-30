# src/scripts/learning_rate_search.py
#
# W&B Bayesian sweep over (eta_c, eta_s) for FedAvg — with or without DP noise.
# Without --dp: calibrates learning_rate and server_lr (used by non-DP scripts).
# With    --dp: calibrates dp_learning_rate and dp_server_lr (used by all DP scripts).
#
# Steps:
#   1. python src/scripts/learning_rate_search.py --config cifar10 [--dp] --create_sweep
#   2. python src/scripts/learning_rate_search.py --config cifar10 [--dp] --sweep_id <id>
#   3. python src/scripts/learning_rate_search.py --config cifar10 [--dp] --write_best --sweep_id <id>

import re
import sys
import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import wandb
from pathlib import Path
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.registry import get_model
from utils.fl_utils  import load_config, get_device, load_data, select_clients, eval_model, apply_aggregate

# Non-DP
SWEEP_CONFIG_NODP = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "eta_c": {"distribution": "log_uniform_values", "min": 0.01,  "max": 1.0},
        "eta_s": {"distribution": "log_uniform_values", "min": 0.1,   "max": 10.0},
    },
    "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 3},
}

# DP
SWEEP_CONFIG_DP = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "eta_c": {"distribution": "log_uniform_values", "min": 0.001, "max": 0.1},
        "eta_s": {"distribution": "log_uniform_values", "min": 0.1,   "max": 3.0},
    },
    "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 3},
}

def run_trial(config, client_datasets, val_loader, dataset_name, rounds, dp):
    mode = "dp-lr-search" if dp else "lr-search"
    wandb.init(group=f"{dataset_name}/{mode}")
    eta_c = wandb.config.eta_c
    eta_s = wandb.config.eta_s

    device    = get_device()
    criterion = nn.CrossEntropyLoss()
    model     = get_model(dataset_name)().to(device)

    K       = config['federated_learning']['clients_per_round']
    N       = config['federated_learning']['num_clients']
    c_fixed = config['differential_privacy']['max_clipping_norm']
    sigma   = config['differential_privacy']['min_noise_multiplier']
    E       = config['federated_learning']['local_epochs']
    bs      = config['federated_learning']['batch_size']

    wandb.define_metric('round')
    wandb.define_metric('*', step_metric='round')

    for t in range(rounds):
        selected   = select_clients(t, N, K)
        raw_deltas = []

        for cid in selected:
            local = copy.deepcopy(model)
            local.train()
            loader = DataLoader(client_datasets[int(cid)], batch_size=bs, shuffle=True)
            for _ in range(E):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    for p in local.parameters():
                        if p.grad is not None:
                            p.grad.detach_()
                            p.grad.zero_()
                    criterion(local(x), y).backward()
                    with torch.no_grad():
                        for p in local.parameters():
                            if p.grad is not None:
                                p.data.sub_(eta_c * p.grad)

            with torch.no_grad():
                dw = torch.cat([
                    (lp.data - gp.data).view(-1).cpu()
                    for lp, gp in zip(local.parameters(), model.parameters())
                ])
            raw_deltas.append(dw)

        with torch.no_grad():
            if dp:
                clipped = [dw * min(1.0, c_fixed / (dw.norm(2).item() + 1e-8)) for dw in raw_deltas]
                agg = torch.stack(clipped).sum(dim=0)
                agg.add_(torch.randn_like(agg) * sigma * c_fixed)
                agg.div_(K)
            else:
                agg = torch.stack(raw_deltas).mean(dim=0)

        apply_aggregate(model, (eta_s * agg).to(device))

        loss, acc = eval_model(model, val_loader, criterion, device)
        wandb.log({"round": t + 1, "val_acc": acc, "val_loss": loss})

        if not torch.isfinite(torch.tensor(loss)):
            wandb.log({"val_acc": 0.0})
            break

    wandb.finish()

def write_best_to_config(config, config_name, sweep_id, dp):
    entity  = config['wandb']['entity']
    project = config['wandb']['project_name']

    api      = wandb.Api()
    sweep    = api.sweep(f"{entity}/{project}/{sweep_id}")
    best_run = sweep.best_run()

    eta_c = best_run.config['eta_c']
    eta_s = best_run.config['eta_s']

    config_path = ROOT / 'src' / 'configs' / f'{config_name}.yaml'
    content     = config_path.read_text()

    if dp:
        content = re.sub(r'(dp_learning_rate:\s*)\S+', rf'\g<1>{eta_c}', content)
        content = re.sub(r'(dp_server_lr:\s*)\S+',     rf'\g<1>{eta_s}', content)
        lr_key, server_key = 'dp_learning_rate', 'dp_server_lr'
    else:
        content = re.sub(r'(learning_rate:\s*)[0-9.e+-]+', rf'\g<1>{eta_c}', content)
        content = re.sub(r'(server_lr:\s*)[0-9.e+-]+',     rf'\g<1>{eta_s}', content)
        lr_key, server_key = 'learning_rate', 'server_lr'

    config_path.write_text(content)

    print(f"Best run: {best_run.name}")
    print(f"  val_acc    = {best_run.summary.get('val_acc', 'n/a'):.4f}")
    print(f"  {lr_key:<20} = {eta_c}")
    print(f"  {server_key:<20} = {eta_s}")
    print(f"Written to {config_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',       type=str, efault='cifar10')
    parser.add_argument('--rounds',       type=int, default=None)
    parser.add_argument('--dp',           action='store_true')
    parser.add_argument('--create_sweep', action='store_true')
    parser.add_argument('--write_best',   action='store_true')
    parser.add_argument('--sweep_id',     type=str, default=None)
    args = parser.parse_args()

    config_path = ROOT / 'src' / 'configs' / f'{args.config}.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"No config found at: {config_path}")
    config = load_config(config_path)

    rounds = args.rounds or (150 if args.dp else 300)
    sweep_config = SWEEP_CONFIG_DP if args.dp else SWEEP_CONFIG_NODP
    mode_label   = "DP" if args.dp else "non-DP"

    if args.create_sweep:
        sweep_id = wandb.sweep(
            sweep_config,
            project=config['wandb']['project_name'],
            entity=config['wandb']['entity'],
        )
        flag = " --dp" if args.dp else ""
        print(f"\n{mode_label} sweep created: {sweep_id}")
        print(f"Run agents with:")
        print(f"  python src/scripts/learning_rate_search.py --config {args.config}{flag} --sweep_id {sweep_id}")
        return

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    client_datasets, val_loader = load_data(config, args.config)

    wandb.agent(
        args.sweep_id,
        function=lambda: run_trial(config, client_datasets, val_loader, args.config, rounds, args.dp),
        project=config['wandb']['project_name'],
        entity=config['wandb']['entity'],
    )

if __name__ == '__main__':
    main()
