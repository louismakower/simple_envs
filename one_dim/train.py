import argparse
import dataclasses
from datetime import datetime
import json
import os
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from louis_rl.ppo import PPORunnerCfg
from louis_rl.sac import SACRunnerCfg
from louis_rl.algorithm import RLRunner
from one_dim.environment import OneDimVecEnv
from one_dim import agents, config


def apply_her_overrides(agent_cfg, args):
    """Override the agent's HER config from CLI flags (used by the sweep launcher)."""
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
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--agent", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument(
            "--headless",
            action="store_true",
            default=False,
            help="Force display off at all times.",
        )
    parser.add_argument("--seed", type=int, default=None,
                        help="If set, seed numpy/torch/random for a reproducible run.")
    parser.add_argument("--her-mode", type=str, default=None, choices=["future", "final"],
                        help="Override the HER goal-sampling mode.")
    parser.add_argument("--her-k", type=int, default=None,
                        help="Override the HER relabeling count k.")
    parser.add_argument("--disable-her", action="store_true", default=False,
                        help="Train without HER (sets her_cfg=None).")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Log subdirectory name; defaults to a timestamp.")
    parser.add_argument("--no-visualise", action="store_true", default=False,
                        help="Skip attaching the matplotlib visualisers.")
    args = parser.parse_args()

    # Seed before constructing the env: OneDimVecEnv.__init__ draws from np.random
    # when ENVIRONMENT_TYPE == "random".
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    env = OneDimVecEnv(num_envs=args.num_envs, max_episode_length=config.EPISODE_LENGTH, headless=args.headless)

    if args.agent == "ppo":
        agent_cfg = agents.PPO_CFG
    elif args.agent =="sac":
        agent_cfg = agents.SAC_CFG

    agent_cfg = apply_her_overrides(agent_cfg, args)

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

    # Attach live matplotlib visualisers over the (state, goal) unit square.
    # The policy heatmap works for both SAC and PPO; the value heatmap needs a
    # critic, so it is SAC-only. Both are skipped when the display is off or
    # when `--no-visualise` is passed (e.g. for parallel sweep runs).
    if not args.headless and not args.no_visualise:
        from one_dim.visualise import Visualiser, ValueVisualiser
        env._visualisers.append(Visualiser(runner, env))
        if args.agent == "sac":
            env._visualisers.append(ValueVisualiser(runner, env))

    runner.learn()
    writer.close()

    # Completion marker — lets the sweep launcher skip finished runs on resume.
    open(os.path.join(log_dir, "DONE"), "w").close()


if __name__ == "__main__":
    main()
