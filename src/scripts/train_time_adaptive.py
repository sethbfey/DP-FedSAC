# src/scripts/train_time_adaptive.py

# $ python src/scripts/train_time_adaptive.py
#     --config   [STR,   default="femnist"]
#     --sigma_t  [FLOAT, default=2.0]
#     --K_save   [INT,   default=50]
#     --T_n_frac [FLOAT, default=0.5]

import sys
import argparse
import torch
import torch.nn as nn
import wandb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.registry import get_model
from utils.fl_utils  import load_config, get_device, load_data, select_clients, local_train, eval_model, apply_aggregate
from utils.rdp       import rdp_per_round

def run_time_adaptive(config, client_datasets, val_loader, sigma_t, K_save, T_n_frac, dataset_name):
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

    T_n    = int(T * T_n_frac)
    q      = K_save / N
    q_full = K / N

    delta_eps_save  = rdp_per_round(int(rdp_alpha), sigma_t, q)
    delta_eps_spend = rdp_per_round(int(rdp_alpha), sigma_t, q_full)
    accum_eps       = 0.0
    momentum_buf    = None

    wandb.init(
        project=config['wandb']['project_name'],
        entity=config['wandb']['entity'],
        group=f'{dataset_name}/time-adaptive',
        name=f'sigma={sigma_t}_K_save={K_save}',
        config={
            'N': N, 'K': K, 'T': T,
            'E':               config['federated_learning']['local_epochs'],
            'eta_c':           config['federated_learning']['learning_rate'],
            'eta_s':           server_lr,
            'server_momentum': server_mo,
            'batch_size':      config['federated_learning']['batch_size'],
            'C':               c_fixed,
            'sigma_t':         sigma_t,
            'K_save':          K_save,
            'T_n':             T_n,
            'T_n_frac':        T_n_frac,
            'rdp_alpha':       rdp_alpha,
            'max_epsilon':     max_eps,
        }
    )
    wandb.define_metric('round')
    wandb.define_metric('*', step_metric='round')

    loss, acc = eval_model(model, val_loader, criterion, device)
    print(f"[TimeAdaptive σ={sigma_t} K_save={K_save} T_n={T_n}] round=0  "
          f"loss={loss:.4f}  val_acc={acc:.4f}  "
          f"δε_save={delta_eps_save:.6f}  δε_spend={delta_eps_spend:.6f}")

    final_round = T
    for t in range(T):
        in_saving     = t < T_n
        k_this_round  = K_save if in_saving else K
        delta_eps     = delta_eps_save if in_saving else delta_eps_spend

        selected   = select_clients(t, N, k_this_round)
        raw_deltas = [local_train(model, client_datasets[int(cid)], config, criterion, device)
                      for cid in selected]

        raw_norms = torch.stack([dw.norm(2) for dw in raw_deltas])
        f_clip    = float((raw_norms > c_fixed).float().mean().item())
        m_norm    = float(raw_norms.median().item())

        with torch.no_grad():
            clipped = [dw * min(1.0, c_fixed / (dw.norm(2).item() + 1e-8)) for dw in raw_deltas]
            agg = torch.stack(clipped).sum(dim=0)
            agg.add_(torch.randn_like(agg) * sigma_t * c_fixed)
            agg.div_(k_this_round)
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
            'K_t':           k_this_round,
        })

        print(f"[TimeAdaptive] round={t+1:>4}  phase={'save ' if in_saving else 'spend'}  "
              f"K={k_this_round}  loss={loss:.4f}  val_acc={acc:.4f}  "
              f"epsilon={accum_eps:.3f}/{max_eps}  "
              f"f_clip={f_clip:.3f}  m_norm={m_norm:.4f}")

        if accum_eps >= max_eps:
            final_round = t + 1
            print(f"\n[TimeAdaptive] Budget exhausted at round {final_round}.")
            break

    wandb.summary['final_val_acc']  = acc
    wandb.summary['final_val_loss'] = loss
    wandb.summary['final_epsilon']  = accum_eps
    wandb.summary['total_rounds']   = final_round
    wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',    type=str,   default='femnist')
    parser.add_argument('--sigma_t',   type=float, default=2.0)
    parser.add_argument('--K_save',    type=int,   default=50)
    parser.add_argument('--T_n_frac',  type=float, default=0.5)
    args = parser.parse_args()

    config_path = ROOT / 'src' / 'configs' / f'{args.config}.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"No config found at: {config_path}")

    config = load_config(config_path)
    client_datasets, val_loader = load_data(config, args.config)

    run_time_adaptive(
        config=config,
        client_datasets=client_datasets,
        val_loader=val_loader,
        sigma_t=args.sigma_t,
        K_save=args.K_save,
        T_n_frac=args.T_n_frac,
        dataset_name=args.config,
    )

if __name__ == '__main__':
    main()
