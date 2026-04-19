# src/scripts/train_dp_fedavg.py

# $ python src/scripts/train_dp_fedavg.py
#     --config  [STR,   default="femnist"]
#     --sigma_t [FLOAT, default=2.0]
#     --seed    [INT,   default=0]

import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import wandb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.registry import get_model
from utils.fl_utils  import load_config, get_device, load_data, select_clients, local_train, eval_model, apply_aggregate
from utils.rdp       import rdp_per_round

def run_dp_fedavg(config, client_datasets, val_loader, sigma_t, dataset_name, seed):
    device    = get_device()
    criterion = nn.CrossEntropyLoss()
    model     = get_model(dataset_name)().to(device)

    K         = config['federated_learning']['clients_per_round']
    N         = config['federated_learning']['num_clients']
    T         = config['federated_learning']['num_global_steps']
    c_fixed   = config['differential_privacy']['max_clipping_norm']
    server_lr = config['federated_learning']['server_lr']
    server_mo = config['federated_learning']['server_momentum']
    rdp_alpha = config['differential_privacy']['rdp_alpha']
    max_eps   = config['differential_privacy']['max_epsilon']

    q           = K / N
    delta_eps   = rdp_per_round(int(rdp_alpha), sigma_t, q)
    accum_eps   = 0.0
    momentum_buf = None

    wandb.init(
        project=config['wandb']['project_name'],
        entity=config['wandb']['entity'],
        group=f'{dataset_name}/dp-fedavg',
        name=f'sigma={sigma_t}_seed={seed}',
        config={
            'N': N, 'K': K, 'T': T,
            'E':               config['federated_learning']['local_epochs'],
            'eta_c':           config['federated_learning']['learning_rate'],
            'eta_s':           server_lr,
            'server_momentum': server_mo,
            'batch_size':      config['federated_learning']['batch_size'],
            'C':               c_fixed,
            'sigma_t':         sigma_t,
            'rdp_alpha':       rdp_alpha,
            'max_epsilon':     max_eps,
            'seed':            seed,
        }
    )
    wandb.define_metric('round')
    wandb.define_metric('*', step_metric='round')

    loss, acc = eval_model(model, val_loader, criterion, device)
    print(f"[DP-FedAvg sigma={sigma_t}] round=0  loss={loss:.4f}  val_acc={acc:.4f}  "
          f"delta_eps/round={delta_eps:.4f}  budget exhausted at round ~{int(max_eps / delta_eps)}")

    final_round = T
    for t in range(T):
        selected   = select_clients(t, N, K)
        raw_deltas = [local_train(model, client_datasets[int(cid)], config, criterion, device)
                      for cid in selected]

        raw_norms = torch.stack([dw.norm(2) for dw in raw_deltas])
        f_clip    = float((raw_norms > c_fixed).float().mean().item())
        m_norm    = float(raw_norms.median().item())

        with torch.no_grad():
            clipped = [dw * min(1.0, c_fixed / (dw.norm(2).item() + 1e-8)) for dw in raw_deltas]
            agg = torch.stack(clipped).sum(dim=0)
            agg.add_(torch.randn_like(agg) * sigma_t * c_fixed)
            agg.div_(K)
            if momentum_buf is None: momentum_buf = agg.clone()
            else:                    momentum_buf.mul_(server_mo).add_(agg)

        apply_aggregate(model, server_lr * momentum_buf)
        accum_eps += delta_eps
        loss, acc = eval_model(model, val_loader, criterion, device)

        wandb.log({
            'round':         t + 1,
            'val_acc':       acc,
            'val_loss':      loss,
            'epsilon':       accum_eps,
            'delta_epsilon': delta_eps,
            'f_clip':        f_clip,
            'm_norm':        m_norm,
        })

        print(f"[DP-FedAvg sigma={sigma_t}] round={t+1:>4}  "
              f"loss={loss:.4f}  val_acc={acc:.4f}  "
              f"epsilon={accum_eps:.3f}/{max_eps}  "
              f"f_clip={f_clip:.3f}  m_norm={m_norm:.4f}")

        if accum_eps >= max_eps:
            final_round = t + 1
            print(f"\n[DP-FedAvg] Budget exhausted at round {final_round}.")
            break

    wandb.summary['final_val_acc']  = acc
    wandb.summary['final_val_loss'] = loss
    wandb.summary['final_epsilon']  = accum_eps
    wandb.summary['total_rounds']   = final_round
    wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  type=str,   default='femnist')
    parser.add_argument('--sigma_t', type=float, default=2.0)
    parser.add_argument('--seed',    type=int,   default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config_path = ROOT / 'src' / 'configs' / f'{args.config}.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"No config found at: {config_path}")

    config = load_config(config_path)
    client_datasets, val_loader = load_data(config, args.config)

    run_dp_fedavg(
        config=config,
        client_datasets=client_datasets,
        val_loader=val_loader,
        sigma_t=args.sigma_t,
        dataset_name=args.config,
        seed=args.seed,
    )

if __name__ == '__main__':
    main()
