import argparse
import dataclasses
from datetime import datetime
import json
import os

from torch.utils.tensorboard import SummaryWriter

from louis_rl.ppo import PPORunner, PPORunnerCfg
from one_dim.environment import OneDimVecEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-episode-length", type=int, default=200)
    parser.add_argument("--num-iterations", type=int, default=2000)
    parser.add_argument("--steps-per-rollout", type=int, default=64)
    parser.add_argument("--num-policy-grad-steps", type=int, default=10)
    parser.add_argument("--num-v-grad-steps", type=int, default=10)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--v-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--experiment-name", type=str, default="ppo_1d")
    args = parser.parse_args()

    log_dir = os.path.join(args.log_dir, args.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    env = OneDimVecEnv(num_envs=args.num_envs, max_episode_length=args.max_episode_length)
    writer = SummaryWriter(log_dir=log_dir)
    cfg = PPORunnerCfg(
        experiment_name=args.experiment_name,
        num_iterations=args.num_iterations,
        steps_per_rollout=args.steps_per_rollout,
        num_policy_grad_steps=args.num_policy_grad_steps,
        num_v_grad_steps=args.num_v_grad_steps,
        policy_lr=args.policy_lr,
        v_lr=args.v_lr,
        gamma=args.gamma,
        eps=args.eps,
        save_interval=args.save_interval,
        policy_hidden_dims=[16],
        v_hidden_dims=[16],
    )
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "cfg.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    runner = PPORunner(env=env, cfg=cfg, log_dir=log_dir, writer=writer)
    runner.learn()
    writer.close()


if __name__ == "__main__":
    main()
