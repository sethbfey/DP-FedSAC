# src/scripts/train_dp_scaffold.py

# $ python src/scripts/train_dp_scaffold.py
#     --config  [STR,   default="femnist"]
#     --sigma_t [FLOAT, default=2.0]

import sys
import copy
import argparse
import torch
import torch.nn as nn
import wandb
from pathlib import Path
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.registry import get_model
from utils.fl_utils  import load_config, get_device, load_data, select_clients, eval_model, apply_aggregate
from utils.rdp       import rdp_per_round

def local_train_scaffold(global_model, dataset, config, criterion, device, client_c, server_c):
    local_model = copy.deepcopy(global_model)
    local_model.train()

    eta_c  = config['federated_learning']['learning_rate']
    E      = config['federated_learning']['local_epochs']
    loader = DataLoader(
        dataset,
        batch_size=config['federated_learning']['batch_size'],
        shuffle=True,
    )

    c_t  = server_c.to(device)
    c_i  = client_c.to(device)

    actual_steps = 0
    for _ in range(E):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            for p in local_model.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

            criterion(local_model(x), y).backward()

            # SCAFFOLD correction: g_i <- g_i + c^t - c_i
            offset = 0
            for p in local_model.parameters():
                numel      = p.numel()
                correction = (c_t[offset:offset+numel] - c_i[offset:offset+numel]).view_as(p.grad)
                p.grad.data.add_(correction)
                offset += numel

            # Regular SGD step
            with torch.no_grad():
                for p in local_model.parameters():
                    p.data.sub_(eta_c * p.grad.data)

            actual_steps += 1

    # Δ_w = θ_local - θ_global
    with torch.no_grad():
        delta_w = torch.cat([
            (lp.data - gp.data).view(-1).cpu()
            for lp, gp in zip(local_model.parameters(), global_model.parameters())
        ])

    return delta_w, actual_steps

def run_dp_scaffold(config, client_datasets, val_loader, sigma_t, dataset_name):
    device     = get_device()
    criterion  = nn.CrossEntropyLoss()
    model      = get_model(dataset_name)().to(device)
    num_params = sum(p.numel() for p in model.parameters())

    K         = config['federated_learning']['clients_per_round']
    N         = config['federated_learning']['num_clients']
    T         = config['federated_learning']['num_global_steps']
    C_w       = config['differential_privacy']['max_clipping_norm']
    E         = config['federated_learning']['local_epochs']
    eta_c     = config['federated_learning']['learning_rate']
    server_lr = config['federated_learning']['server_lr']
    rdp_alpha = config['differential_privacy']['rdp_alpha']
    max_eps   = config['differential_privacy']['max_epsilon']

    C_c = C_w / (E * eta_c)  # control variate clipping norm, derived from SCAFFOLD's drift correction term

    q = K / N
    delta_eps_per_round = rdp_per_round(int(rdp_alpha), sigma_t, q)

    accum_eps = 0.0

    server_c        = torch.zeros(num_params)
    client_controls = {}

    wandb.init(
        project=config['wandb']['project_name'],
        entity=config['wandb']['entity'],
        group=f'{dataset_name}/dp-scaffold',
        name=f'sigma={sigma_t}',
        config={
            'N': N, 'K': K, 'T': T,
            'E':               config['federated_learning']['local_epochs'],
            'eta_c':           config['federated_learning']['learning_rate'],
            'eta_s':           server_lr,
            'server_momentum': 0.0,   # SCAFFOLD handles drift via control variates
            'batch_size':      config['federated_learning']['batch_size'],
            'C_w':             C_w,
            'sigma_t':         sigma_t,
            'rdp_alpha':       rdp_alpha,
            'max_epsilon':     max_eps,
            'delta_eps_per_round': delta_eps_per_round,
        }
    )
    wandb.define_metric('round')
    wandb.define_metric('*', step_metric='round')

    loss, acc = eval_model(model, val_loader, criterion, device)
    print(f"[DP-SCAFFOLD sigma={sigma_t}] round=0  loss={loss:.4f}  val_acc={acc:.4f}  "
          f"delta_eps/round={delta_eps_per_round:.4f}  "
          f"budget exhausted at round ~{int(max_eps / delta_eps_per_round)}")

    final_round = T
    for t in range(T):
        selected = select_clients(t, N, K)

        raw_delta_w = []
        raw_delta_c = []

        for cid in selected:
            cid = int(cid)

            if cid not in client_controls:
                client_controls[cid] = torch.zeros(num_params)

            dw, actual_steps = local_train_scaffold(
                model, client_datasets[cid], config, criterion, device,
                client_controls[cid], server_c,
            )

            clipped_dw_for_c = dw * min(1.0, C_w / (dw.norm(2).item() + 1e-8))  # keeps c_i bounded
            dc = -server_c.cpu() - clipped_dw_for_c * (1.0 / (actual_steps * eta_c))

            client_controls[cid] = client_controls[cid] + dc

            raw_delta_w.append(dw)
            raw_delta_c.append(dc)

        raw_norms_w = torch.stack([dw.norm(2) for dw in raw_delta_w])
        f_clip      = float((raw_norms_w > C_w).float().mean())
        m_norm      = float(raw_norms_w.median())

        with torch.no_grad():
            # Clip Δ_w to C_w (model updates)
            clipped_w = [dw * min(1.0, C_w / (dw.norm(2).item() + 1e-8)) for dw in raw_delta_w]
            # Clip Δ_c to C_c (control updates)
            clipped_c = [dc * min(1.0, C_c / (dc.norm(2).item() + 1e-8)) for dc in raw_delta_c]

            # Model update aggregation
            agg_w = torch.stack(clipped_w).sum(dim=0)
            agg_w.add_(torch.randn_like(agg_w) * sigma_t * C_w)
            agg_w.div_(K)
            agg_c = torch.stack(clipped_c).sum(dim=0)
            agg_c.div_(K)

            # θ^(t+1) = θ^t + η_s · agg_w
            apply_aggregate(model, (server_lr * agg_w).to(device))

            # c^(t+1) = c^t + (K/N) · agg_c
            server_c.add_((K / N) * agg_c)

        accum_eps += delta_eps_per_round
        loss, acc  = eval_model(model, val_loader, criterion, device)

        wandb.log({
            'round':         t + 1,
            'val_acc':       acc,
            'val_loss':      loss,
            'epsilon':       accum_eps,
            'delta_epsilon': delta_eps_per_round,
            'f_clip':        f_clip,
            'm_norm':        m_norm,
        })

        print(f"[DP-SCAFFOLD sigma={sigma_t}] round={t+1:>4}  "
              f"loss={loss:.4f}  val_acc={acc:.4f}  "
              f"epsilon={accum_eps:.3f}/{max_eps}  "
              f"f_clip={f_clip:.3f}  m_norm={m_norm:.4f}")

        if accum_eps >= max_eps:
            final_round = t + 1
            print(f"\n[DP-SCAFFOLD] Budget exhausted at round {final_round}.")
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
    args = parser.parse_args()

    config_path = ROOT / 'src' / 'configs' / f'{args.config}.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"No config found at: {config_path}")

    config = load_config(config_path)
    client_datasets, val_loader = load_data(config, args.config)

    run_dp_scaffold(
        config=config,
        client_datasets=client_datasets,
        val_loader=val_loader,
        sigma_t=args.sigma_t,
        dataset_name=args.config,
    )


if __name__ == '__main__':
    main()
