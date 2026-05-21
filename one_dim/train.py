import argparse
import dataclasses
from datetime import datetime
import json
import os

from torch.utils.tensorboard import SummaryWriter

from louis_rl.ppo import PPORunnerCfg
from louis_rl.sac import SACRunnerCfg
from louis_rl.algorithm import RLRunner
from one_dim.environment import OneDimVecEnv
from one_dim import agents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--agent", type=str, required=True)
    parser.add_argument("--max-episode-length", type=int, default=200)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument(
            "--headless",
            action="store_true",
            default=False,
            help="Force display off at all times.",
        )
    args = parser.parse_args()


    env = OneDimVecEnv(num_envs=args.num_envs, max_episode_length=args.max_episode_length, headless=args.headless)

    if args.agent == "ppo":
        agent_cfg = agents.PPO_CFG
    elif args.agent =="sac":
        agent_cfg = agents.SAC_CFG
    
    log_dir = os.path.join(args.log_dir, agent_cfg.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)

    
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "cfg.json"), "w") as f:
        json.dump(dataclasses.asdict(agent_cfg), f, indent=2)

    runner = RLRunner(env=env, cfg=agent_cfg, log_dir=log_dir, writer=writer)
    runner.learn()
    writer.close()


if __name__ == "__main__":
    main()
