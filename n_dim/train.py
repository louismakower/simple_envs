import argparse
import dataclasses
from datetime import datetime
import json
import os
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from louis_rl.algorithm import RLRunner
from n_dim import agents
from n_dim.env import NDimVecEnv
from n_dim.env_cfg import NDimVecEnvCfg


def apply_her_overrides(agent_cfg, args):
    if args.disable_her:
        return dataclasses.replace(agent_cfg, her_cfg=None)
    if args.her_mode is None and args.her_k is None:
        return agent_cfg
    her_cfg = agent_cfg.her_cfg
    if her_cfg is None:
        raise ValueError("--her-mode/--her-k given but the agent config has no her_cfg")
    overrides = {}
    if args.her_mode is not None:
        overrides["mode"] = args.her_mode
    if args.her_k is not None:
        overrides["k"] = args.her_k
    return dataclasses.replace(agent_cfg, her_cfg=dataclasses.replace(her_cfg, **overrides))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True, choices=["ppo", "sac"])
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--n", type=int, default=None, help="State / action dimensionality.")
    parser.add_argument("--max_ep_len", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--her_mode", type=str, default=None, choices=["future", "final"])
    parser.add_argument("--her_k", type=int, default=None)
    parser.add_argument("--disable_her", action="store_true", default=False)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--no_visualise", action="store_true", default=False)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    env_overrides = {"num_envs": args.num_envs}
    if args.n is not None:
        env_overrides["n"] = args.n
    if args.max_ep_len is not None:
        env_overrides["max_ep_len"] = args.max_ep_len
    env_cfg = dataclasses.replace(NDimVecEnvCfg(), **env_overrides)
    env = NDimVecEnv(env_cfg)

    if args.agent == "ppo":
        agent_cfg = agents.PPO_CFG
    else:
        agent_cfg = agents.SAC_CFG

    if agent_cfg.her_cfg is not None:
        her_cfg = dataclasses.replace(agent_cfg.her_cfg, n=env_cfg.n, goal_radius=env_cfg.goal_radius)
        agent_cfg = dataclasses.replace(agent_cfg, her_cfg=her_cfg)

    agent_cfg = apply_her_overrides(agent_cfg, args)
    print(agent_cfg)

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name is not None:
        log_dir = os.path.join(args.log_dir, run_name)
    else:
        log_dir = os.path.join(args.log_dir, agent_cfg.experiment_name, run_name)
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)

    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "cfg.json"), "w") as f:
        json.dump(dataclasses.asdict(agent_cfg), f, indent=2)

    runner = RLRunner(env=env, cfg=agent_cfg, log_dir=log_dir, writer=writer)

    if not args.no_visualise:
        from n_dim.visualise import PolicyVisualiser, SACValueVisualiser, PPOValueVisualiser
        env._visualisers.append(PolicyVisualiser(runner, env))
        if args.agent == "sac":
            env._visualisers.append(SACValueVisualiser(runner, env))
        else:
            env._visualisers.append(PPOValueVisualiser(runner, env))

    runner.learn()
    writer.close()

    open(os.path.join(log_dir, "DONE"), "w").close()


if __name__ == "__main__":
    main()
