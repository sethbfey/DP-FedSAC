# src/scripts/train_fedavg.py

# $ python src/scripts/train_fedavg.py
#     --config       [STR,   default="femnist"]
#     --es_patience  [INT,   default=200]
#     --es_min_delta [FLOAT, default=1e-4]

import re
import sys
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

def write_max_clipping_norm_to_config(config_path: Path, p75):
    content = config_path.read_text()
    content = re.sub(r'(max_clipping_norm:\s*)[0-9.]+', rf'\g<1>{p75:.4f}', content)
    config_path.write_text(content)

def run_fedavg(config, config_path, client_datasets, val_loader, dataset_name,
               es_patience=200, es_min_delta=1e-4):
    device    = get_device()
    criterion = nn.CrossEntropyLoss()
    model     = get_model(dataset_name)().to(device)

    K         = config['federated_learning']['clients_per_round']
    N         = config['federated_learning']['num_clients']
    T         = config['federated_learning']['num_global_steps']
    server_lr = config['federated_learning']['server_lr']
    server_mo = config['federated_learning']['server_momentum']

    all_norms      = []
    best_val_loss  = float('inf')
    patience_count = 0
    momentum_buf   = None

    wandb.init(
        project=config['wandb']['project_name'],
        entity=config['wandb']['entity'],
        group=f'{dataset_name}/fedavg',
        name='fedavg',
        config={
            'N': N, 'K': K, 'T': T,
            'E':               config['federated_learning']['local_epochs'],
            'eta_c':           config['federated_learning']['learning_rate'],
            'eta_s':           server_lr,
            'server_momentum': server_mo,
            'batch_size':      config['federated_learning']['batch_size'],
        }
    )
    wandb.define_metric('round')
    wandb.define_metric('*', step_metric='round')

    loss, acc = eval_model(model, val_loader, criterion, device)
    print(f"[FedAvg] round=0  loss={loss:.4f}  val_acc={acc:.4f}")

    for t in range(T):
        selected   = select_clients(t, N, K)
        raw_deltas = [local_train(model, client_datasets[int(cid)], config, criterion, device)
                      for cid in selected]

        raw_norms_t = [dw.norm(2).item() for dw in raw_deltas]
        all_norms.extend(raw_norms_t)

        with torch.no_grad():
            agg = torch.stack(raw_deltas).mean(dim=0)
            if momentum_buf is None: momentum_buf = agg.clone()
            else:                    momentum_buf.mul_(server_mo).add_(agg)
        apply_aggregate(model, server_lr * momentum_buf)

        loss, acc = eval_model(model, val_loader, criterion, device)

        wandb.log({
            'round':         t + 1,
            'val_acc':       acc,
            'val_loss':      loss,
            'grad_norm_p50': float(np.percentile(raw_norms_t, 50)),
            'grad_norm_p75': float(np.percentile(raw_norms_t, 75)),
        })

        if (t + 1) % 50 == 0:
            print(f"[FedAvg] round={t+1:>4}  loss={loss:.4f}  val_acc={acc:.4f}  "
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

    norms_array = np.array(all_norms, dtype=np.float32)
    p75 = float(np.percentile(norms_array, 75))

    print(f"\n[FedAvg] Gradient norm percentiles across all rounds:")
    for pct in [50, 75, 90, 95]:
        print(f"  p{pct}: {np.percentile(norms_array, pct):.4f}")
    print(f"\n  -> C_max = p75 = {p75:.4f}; writing to {config_path}")

    wandb.summary['c_max']          = p75
    wandb.summary['final_val_acc']  = acc
    wandb.summary['final_val_loss'] = loss
    wandb.summary['total_rounds']   = t + 1
    wandb.finish()

    write_max_clipping_norm_to_config(config_path, p75)
    print(f"  {config_path.name} max_clipping_norm updated to {p75:.4f}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',       type=str,   default='femnist')
    parser.add_argument('--es_patience',  type=int,   default=200)
    parser.add_argument('--es_min_delta', type=float, default=1e-4)
    args = parser.parse_args()

    config_path = ROOT / 'src' / 'configs' / f'{args.config}.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"No config found at: {config_path}")

    config = load_config(config_path)
    client_datasets, val_loader = load_data(config, args.config)

    run_fedavg(
        config=config,
        config_path=config_path,
        client_datasets=client_datasets,
        val_loader=val_loader,
        dataset_name=args.config,
        es_patience=args.es_patience,
        es_min_delta=args.es_min_delta,
    )

if __name__ == '__main__':
    main()
