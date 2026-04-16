# src/scripts/train_sac.py

# Training (is_training_agent=True, per-round seeds):
# $ python src/scripts/train_sac.py
#     --config              [STR, default="femnist"]
#     --num_episodes        [INT, default=100]
#     --train_rounds        [INT, default=1000]
#     --warmup_episodes     [INT, default=5]
#     --updates_per_step    [INT, default=5]
#     --reward_beta         [FLOAT, default=None]
#     --checkpoint_interval [INT, default=1]
#     --save_path           [STR, default=src/checkpoints/<config>/beta_<beta>/sac_episode_<num_episodes>.pt]
#     --resume              [STR, default=None]

# Evaluation (is_training_agent=False, fixed per-round seeds):
# $ python src/scripts/train_sac.py --eval_only
#     --config        [STR, default="femnist"]
#     --eval_episodes [INT, default=1]
#     --resume        [STR, required]

import sys
import copy
import argparse
import yaml
import numpy as np
import torch
import wandb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from envs.fl_env import FL_DP_Env
from models.sac import SAC

def load_config(config_path: Path):
    with config_path.open('r') as f:
        return yaml.safe_load(f)

def train(train_config, args, agent, start_episode=1, total_steps=0,
          checkpoint_dir=None, run_id=None):

    env          = FL_DP_Env(train_config, is_training_agent=True)
    warmup_steps = args.warmup_episodes * train_config['federated_learning']['num_global_steps']

    for episode in range(start_episode, args.num_episodes + 1):
        obs, _         = env.reset()
        ep_reward      = 0.0
        ep_rounds      = 0
        C_hist         = []
        sigma_hist     = []
        update_metrics = {}
        info           = {}

        done = False
        while not done:
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(obs, action, reward, next_obs, float(done))

            if total_steps >= warmup_steps:
                for _ in range(args.updates_per_step):
                    m = agent.update()
                    if m:
                        update_metrics = m

            obs        = next_obs
            ep_reward += reward
            ep_rounds += 1
            total_steps += 1
            C_hist.append(info['C_t'])
            sigma_hist.append(info['sigma_t'])

        log = {
            'episode':        episode,
            'episode_reward': ep_reward,
            'episode_rounds': ep_rounds,
            'final_accuracy': info['acc_after'],
            'final_epsilon':  env.current_epsilon,
            'mean_C_t':       float(np.mean(C_hist)),
            'mean_sigma_t':   float(np.mean(sigma_hist)),
            'buffer_size':    len(agent.replay_buffer),
            'total_steps':    total_steps,
        }
        log.update(update_metrics)
        wandb.log(log)

        print(f"[ep {episode:>4}/{args.num_episodes}]  "
              f"reward={ep_reward:+.4f}  rounds={ep_rounds}  "
              f"acc={info['acc_after']:.4f}  ε={env.current_epsilon:.3f}  "
              f"mean_C={np.mean(C_hist):.4f}  mean_σ={np.mean(sigma_hist):.4f}  "
              f"α={update_metrics.get('alpha', float('nan')):.4f}")

        if checkpoint_dir and episode % args.checkpoint_interval == 0:
            ckpt_path = checkpoint_dir / f'sac_episode_{episode:04d}.pt'
            agent.save(str(ckpt_path), episode=episode, total_steps=total_steps,
                       wandb_run_id=run_id)
            print(f"  → checkpoint saved: {ckpt_path}")

    return agent

def evaluate(config, agent, num_eval_episodes=3):
    print(f"\n{'='*60}")
    print(f"Final evaluation — {num_eval_episodes} episodes")
    print(f"{'='*60}")

    env       = FL_DP_Env(config, is_training_agent=False)
    acc_list  = []
    eps_list  = []
    rew_list  = []

    for ep in range(1, num_eval_episodes + 1):
        obs, _ = env.reset()
        done      = False
        ep_reward = 0.0
        info      = {}
        C_hist, sigma_hist = [], []

        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            C_hist.append(info['C_t'])
            sigma_hist.append(info['sigma_t'])

        acc_list.append(info['acc_after'])
        eps_list.append(env.current_epsilon)
        rew_list.append(ep_reward)

        print(f"  eval ep {ep}: acc={info['acc_after']:.4f}  ε={env.current_epsilon:.3f}  "
              f"reward={ep_reward:+.4f}  "
              f"mean_C={np.mean(C_hist):.4f}  mean_σ={np.mean(sigma_hist):.4f}")

    mean_acc = float(np.mean(acc_list))
    mean_eps = float(np.mean(eps_list))
    mean_rew = float(np.mean(rew_list))

    print(f"\n  mean_acc={mean_acc:.4f}  mean_ε={mean_eps:.4f}  mean_reward={mean_rew:+.4f}")

    wandb.summary['eval_mean_accuracy'] = mean_acc
    wandb.summary['eval_mean_epsilon']  = mean_eps
    wandb.summary['eval_mean_reward']   = mean_rew

    return mean_acc, mean_eps

def main():
    parser = argparse.ArgumentParser(description='Train SAC agent.')
    parser.add_argument('--config', type=str, default='femnist')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--train_rounds', type=int, default=1000)
    parser.add_argument('--warmup_episodes', type=int, default=5)
    parser.add_argument('--updates_per_step', type=int, default=5)
    parser.add_argument('--reward_beta', type=float, default=None)
    parser.add_argument('--eval_episodes', type=int, default=1)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--checkpoint_interval', type=int, default=1)
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()

    config_path = ROOT / 'src' / 'configs' / f'{args.config}.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"No YAML config found at: {config_path}")
    config = load_config(config_path)

    if args.reward_beta is not None:
        config['sac_agent']['reward_beta'] = args.reward_beta

    train_config = copy.deepcopy(config)
    full_T       = config['federated_learning']['num_global_steps']
    full_eps_max = config['differential_privacy']['max_epsilon']

    if args.train_rounds is not None and args.train_rounds < full_T:
        train_config['federated_learning']['num_global_steps'] = args.train_rounds
        train_config['differential_privacy']['max_epsilon'] = (
            full_eps_max * (args.train_rounds / full_T)
        )
        print(f"Training with {args.train_rounds} rounds/episode "
              f"(ε_max scaled to {train_config['differential_privacy']['max_epsilon']:.4f})")

    beta = config['sac_agent']['reward_beta']

    # Resume support
    saved_run_id = None
    if args.resume:
        probe = torch.load(args.resume, map_location='cpu', weights_only=False)
        saved_run_id = probe.get('wandb_run_id', None)

    run = wandb.init(
        project=config['wandb']['project_name'],
        entity=config['wandb']['entity'],
        name=f'DP-FedSAC-{args.config}-beta{beta}',
        group=f'{args.config}-DP-FedSAC',
        tags=['dp-fedsac', args.config, 'sac'],
        id=saved_run_id,
        resume='allow' if saved_run_id else None,
        config={
            'algorithm':        'DP-FedSAC',
            'dataset':          args.config,
            'reward_beta':      beta,
            'num_episodes':     args.num_episodes,
            'train_rounds':     train_config['federated_learning']['num_global_steps'],
            'warmup_episodes':  args.warmup_episodes,
            'updates_per_step': args.updates_per_step,
            **{f'fl_{k}': v for k, v in config['federated_learning'].items()},
            **{f'dp_{k}': v for k, v in config['differential_privacy'].items()},
            **{f'sac_{k}': v for k, v in config['sac_agent'].items()},
        }
    )
    wandb.define_metric('episode')
    wandb.define_metric('*', step_metric='episode')

    checkpoint_dir = ROOT / 'src' / 'checkpoints' / args.config / f'beta_{beta}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.save_path is None:
        args.save_path = str(checkpoint_dir / f'sac_episode_{args.num_episodes:04d}.pt')

    agent         = SAC(obs_dim=6, action_dim=2, config=config)
    start_episode = 1
    start_steps   = 0

    if args.eval_only:
        if not args.resume: raise ValueError("--eval_only requires --resume <checkpoint path>")
        print(f"Evaluation only — loading checkpoint: {args.resume}")
        agent.load(args.resume)
        evaluate(config, agent, num_eval_episodes=args.eval_episodes)
        wandb.finish()
        return

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_episode, start_steps = agent.load(args.resume)
        start_episode += 1
        print(f"  → resuming at episode {start_episode}, total_steps={start_steps}\n")

    print(f"\nDP-FedSAC training: episodes {start_episode}–{args.num_episodes}, "
          f"{train_config['federated_learning']['num_global_steps']} rounds/episode, "
          f"reward_beta={beta}\n")

    # Training
    agent = train(
        train_config, args, agent,
        start_episode=start_episode,
        total_steps=start_steps,
        checkpoint_dir=checkpoint_dir,
        run_id=run.id,
    )

    # Save final model
    agent.save(args.save_path, episode=args.num_episodes, wandb_run_id=run.id)
    artifact = wandb.Artifact(f'dp-fedsac-femnist-beta{beta}', type='model')
    artifact.add_file(args.save_path)
    run.log_artifact(artifact)
    print(f"\nAgent saved to {args.save_path}")

    # Evaluation
    evaluate(config, agent, num_eval_episodes=args.eval_episodes)
    wandb.finish()


if __name__ == '__main__':
    main()
