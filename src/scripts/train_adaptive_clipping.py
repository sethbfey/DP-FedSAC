# src/scripts/train_adaptive_clipping.py

# $ python src/scripts/train_adaptive_clipping.py
#     --config        [STR,   default="femnist"]
#     --sigma_t       [FLOAT, default=2.0]
#     --gamma_target  [FLOAT, default=0.5]
#     --eta_C         [FLOAT, default=0.2]
#     --sigma_b       [FLOAT, default=K/20]
#     --seed          [INT,   default=0]

import sys
import math
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
from utils.fl_utils  import load_config, get_device, load_data, select_clients, local_train, eval_model, apply_aggregate, get_dp_lr, get_dp_server_lr
from utils.rdp       import rdp_per_round

def run_adaptive_clipping(config, client_datasets, val_loader, sigma_t, gamma_target, eta_C, sigma_b, dataset_name, seed):
    device    = get_device()
    criterion = nn.CrossEntropyLoss()
    model     = get_model(dataset_name)().to(device)

    K         = config['federated_learning']['clients_per_round']
    N         = config['federated_learning']['num_clients']
    T         = config['federated_learning']['num_global_steps']
    C_init    = config['differential_privacy']['max_clipping_norm']
    server_lr = get_dp_server_lr(config)
    server_mo = config['federated_learning']['server_momentum']
    rdp_alpha = config['differential_privacy']['rdp_alpha']
    max_eps   = config['differential_privacy']['max_epsilon']

    q               = K / N
    delta_eps_model = rdp_per_round(int(rdp_alpha), sigma_t, q)
    delta_eps_count = rdp_per_round(int(rdp_alpha), sigma_b, q)
    delta_eps       = delta_eps_model + delta_eps_count
    accum_eps       = 0.0
    momentum_buf    = None

    log_C = math.log(C_init)

    wandb.init(
        project=config['wandb']['project_name'],
        entity=config['wandb']['entity'],
        group=f'{dataset_name}/adaptive-clipping',
        name=f'sigma={sigma_t}_gamma={gamma_target}_etaC={eta_C}_seed={seed}',
        config={
            'N': N, 'K': K, 'T': T,
            'E':               config['federated_learning']['local_epochs'],
            'eta_c':           get_dp_lr(config),
            'eta_s':           server_lr,
            'server_momentum': server_mo,
            'batch_size':      config['federated_learning']['batch_size'],
            'C_init':          C_init,
            'sigma_t':         sigma_t,
            'gamma_target':    gamma_target,
            'eta_C':           eta_C,
            'sigma_b':         sigma_b,
            'rdp_alpha':       rdp_alpha,
            'max_epsilon':     max_eps,
            'delta_eps_model': delta_eps_model,
            'delta_eps_count': delta_eps_count,
            'seed':            seed,
        }
    )
    wandb.define_metric('round')
    wandb.define_metric('*', step_metric='round')

    loss, acc = eval_model(model, val_loader, criterion, device)
    print(f"[AdaptiveClip sigma={sigma_t} gamma={gamma_target} eta_C={eta_C}] round=0  "
          f"loss={loss:.4f}  val_acc={acc:.4f}  "
          f"delta_eps/round={delta_eps:.6f} (model={delta_eps_model:.6f} count={delta_eps_count:.6f})  "
          f"budget exhausted at round ~{int(max_eps / delta_eps)}")

    final_round = T
    for t in range(T):
        C_t        = math.exp(log_C)
        selected   = select_clients(t, N, K)
        raw_deltas = [local_train(model, client_datasets[int(cid)], config, criterion, device, lr=get_dp_lr(config))
                      for cid in selected]
        raw_deltas = [dw for dw in raw_deltas if torch.isfinite(dw).all()]
        n_nan      = K - len(raw_deltas)

        raw_norms = torch.stack([dw.norm(2) for dw in raw_deltas])
        f_clip    = float((raw_norms > C_t).float().mean().item())
        m_norm    = float(raw_norms.median().item())

        # Unclipped indicator b_i = 1[||Δ_w_i|| <= C_t]
        b_sum       = float((raw_norms <= C_t).float().sum().item())
        b_sum_noisy = b_sum + float(np.random.randn() * sigma_b)
        b_hat       = b_sum_noisy / K

        with torch.no_grad():
            clipped = [dw * min(1.0, C_t / (dw.norm(2).item() + 1e-8)) for dw in raw_deltas]
            agg = torch.stack(clipped).sum(dim=0)
            agg.add_(torch.randn_like(agg) * sigma_t * C_t)
            agg.div_(K)
            if momentum_buf is None: momentum_buf = agg.clone()
            else:                    momentum_buf.mul_(server_mo).add_(agg)

        apply_aggregate(model, server_lr * momentum_buf)

        # Geometric update in log space: C_{t+1} = C_t * exp(-eta_C * (b_hat - gamma))
        log_C -= eta_C * (b_hat - gamma_target)

        accum_eps += delta_eps
        loss, acc  = eval_model(model, val_loader, criterion, device)

        wandb.log({
            'round':         t + 1,
            'val_acc':       acc,
            'val_loss':      loss,
            'epsilon':       accum_eps,
            'delta_epsilon': delta_eps,
            'f_clip':        f_clip,
            'm_norm':        m_norm,
            'C_t':           C_t,
            'b_hat':         b_hat,
            'n_nan':         n_nan,
        })

        print(f"[AdaptiveClip sigma={sigma_t}] round={t+1:>4}  "
              f"loss={loss:.4f}  val_acc={acc:.4f}  "
              f"epsilon={accum_eps:.3f}/{max_eps}  "
              f"C_t={C_t:.4f}  b_hat={b_hat:.3f}  "
              f"f_clip={f_clip:.3f}  m_norm={m_norm:.4f}")

        if accum_eps >= max_eps:
            final_round = t + 1
            print(f"\n[AdaptiveClip] Budget exhausted at round {final_round}.")
            break

    wandb.summary['final_val_acc']  = acc
    wandb.summary['final_val_loss'] = loss
    wandb.summary['final_epsilon']  = accum_eps
    wandb.summary['total_rounds']   = final_round
    wandb.summary['final_C_t']      = math.exp(log_C)
    wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',       type=str,   default='femnist')
    parser.add_argument('--sigma_t',      type=float, default=2.0)
    parser.add_argument('--gamma_target', type=float, default=0.5)
    parser.add_argument('--eta_C',        type=float, default=0.2)
    parser.add_argument('--sigma_b',      type=float, default=None)
    parser.add_argument('--seed',         type=int,   default=0)
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

    K = config['federated_learning']['clients_per_round']
    sigma_b = args.sigma_b if args.sigma_b is not None else K / 20.0

    run_adaptive_clipping(
        config=config,
        client_datasets=client_datasets,
        val_loader=val_loader,
        sigma_t=args.sigma_t,
        gamma_target=args.gamma_target,
        eta_C=args.eta_C,
        sigma_b=sigma_b,
        dataset_name=args.config,
        seed=args.seed,
    )

if __name__ == '__main__':
    main()
