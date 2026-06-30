import argparse
import dataclasses
from datetime import datetime
import json
import os
import random

import numpy as np
import torch

from louis_rl.rl_runner import RLRunner
from n_dim import agents as n_dim_agents
from n_dim.env import NDimVecEnv
from n_dim.env_cfg import NDimVecEnvCfg
from walls import agents as walls_agents
from walls.env import WallsVecEnv
from walls.env_cfg import WallsVecEnvCfg
from momentum.env import MomentumWallsVecEnv
from momentum.env_cfg import MomentumVecEnvCfg
from momentum import agents as momentum_agents


# task name -> (env class, env cfg class, agents module)
TASKS = {
    "n_dim": (NDimVecEnv, NDimVecEnvCfg, n_dim_agents),
    "walls": (WallsVecEnv, WallsVecEnvCfg, walls_agents),
    "momentum": (MomentumWallsVecEnv, MomentumVecEnvCfg, momentum_agents)
}


def build_env_overrides(task, args):
    overrides = {"num_envs": args.num_envs}
    if args.max_ep_len is not None:
        overrides["max_ep_len"] = args.max_ep_len
    if args.goal_radius is not None:
        overrides["goal_radius"] = args.goal_radius

    if task == "n_dim":
        if args.n is not None:
            overrides["n"] = args.n
    elif task == "walls":
        if args.n_walls is not None:
            overrides["n_walls"] = args.n_walls
        if args.gap_width is not None:
            overrides["gap_width"] = args.gap_width
        if args.maze_seed is not None:
            overrides["seed"] = args.maze_seed

    return overrides


def build_visualisers(task, agent, runner, env, record):
    if task == "n_dim":
        from n_dim.visualise import (
            PolicyVisualiser, SACValueVisualiser, PPOValueVisualiser, BufferVisualiser,
        )
        visualisers = [PolicyVisualiser(runner, env, record=record)]
        if agent == "sac":
            visualisers.append(SACValueVisualiser(runner, env, record=record))
            visualisers.append(BufferVisualiser(runner, env, record=record))
        else:
            visualisers.append(PPOValueVisualiser(runner, env, record=record))
        return visualisers

    if task == "walls":
        from walls.visualise import (
            WallsPolicyVisualiser, WallsSACValueVisualiser,
            WallsPPOValueVisualiser, WallsBufferVisualiser,
            WallsIntrinsicRewardVisualiser, WallsPPOIntrinsicValueVisualiser,
        )
        visualisers = [WallsPolicyVisualiser(runner, env, record=record)]
        if agent == "sac":
            visualisers.append(WallsSACValueVisualiser(runner, env, record=record))
            visualisers.append(WallsBufferVisualiser(runner, env, record=record))
            if getattr(runner.runner, "intrinsic", None) is not None:
                visualisers.append(WallsIntrinsicRewardVisualiser(runner, env, record=record))
        else:
            visualisers.append(WallsPPOValueVisualiser(runner, env, record=record))
            if getattr(runner.runner, "intrinsic", None) is not None:
                visualisers.append(WallsIntrinsicRewardVisualiser(runner, env, record=record))
                visualisers.append(WallsPPOIntrinsicValueVisualiser(runner, env, record=record))
        return visualisers

    if task == "momentum":
        from momentum.visualise import (
            MomentumPolicyVisualiser, MomentumSACValueVisualiser,
            MomentumPPOValueVisualiser, MomentumBufferVisualiser,
            MomentumIntrinsicRewardVisualiser, MomentumPPOIntrinsicValueVisualiser,
        )
        every = 200
        visualisers = [MomentumPolicyVisualiser(runner, env, update_every=every, record=record)]
        if agent == "sac":
            visualisers.append(MomentumSACValueVisualiser(runner, env, update_every=every, record=record))
            visualisers.append(MomentumBufferVisualiser(runner, env, update_every=every, record=record))
            if getattr(runner.runner, "intrinsic", None) is not None:
                visualisers.append(MomentumIntrinsicRewardVisualiser(runner, env, update_every=every, record=record))
        else:
            visualisers.append(MomentumPPOValueVisualiser(runner, env, update_every=every, record=record))
            if getattr(runner.runner, "intrinsic", None) is not None:
                visualisers.append(MomentumIntrinsicRewardVisualiser(runner, env, update_every=every, record=record))
                visualisers.append(MomentumPPOIntrinsicValueVisualiser(runner, env, update_every=every, record=record))
        return visualisers

    print(f"[INFO] no visualisers for task '{task}'; training without them.")
    return []


def apply_intrinsic_overrides(agent_cfg, agent, mode, agents_module):
    """Override the agent's intrinsic-reward mode from --intrinsic.

    PPO and SAC expose intrinsic differently: PPO has a single ``intrinsic``
    field (None | a cfg), SAC has ``use_intrinsic`` + ``intrinsic_cfg``. We map
    the flag onto whichever the algo uses, pulling the RND/Counts hyperparameters
    from named constants (``<ALGO>_RND_CFG`` / ``<ALGO>_COUNTS_CFG``) on the
    task's agents module. ``mode=None`` leaves the config untouched.
    """
    if mode is None:
        return agent_cfg

    prefix = "PPO" if agent == "ppo" else "SAC"
    sub_cfg = None
    if mode != "none":
        attr = f"{prefix}_{mode.upper()}_CFG"
        sub_cfg = getattr(agents_module, attr, None)
        if sub_cfg is None:
            raise ValueError(
                f"--intrinsic {mode} requested but {agents_module.__name__} "
                f"defines no {attr}"
            )

    if agent == "ppo":
        return dataclasses.replace(agent_cfg, intrinsic=sub_cfg)
    # sac
    if mode == "none":
        return dataclasses.replace(agent_cfg, use_intrinsic=False)
    return dataclasses.replace(agent_cfg, use_intrinsic=True, intrinsic_cfg=sub_cfg)


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
    parser.add_argument("--task", type=str, required=True, choices=list(TASKS))
    parser.add_argument("--agent", type=str, required=True, choices=["ppo", "sac"])
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--max_ep_len", type=int, default=None)
    parser.add_argument("--goal_radius", type=float, default=None)
    # n_dim-specific
    parser.add_argument("--n", type=int, default=None, help="[n_dim] state / action dimensionality.")
    # walls-specific
    parser.add_argument("--n_walls", type=int, default=None, help="[walls] number of walls.")
    parser.add_argument("--gap_width", type=float, default=None, help="[walls] hole height.")
    parser.add_argument("--maze_seed", type=int, default=None, help="[walls] seed for the frozen maze layout.")

    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=None, help="Global RNG seed (python/numpy/torch).")
    parser.add_argument("--her_mode", type=str, default=None, choices=["future", "final"])
    parser.add_argument("--her_k", type=int, default=None)
    parser.add_argument("--disable_her", action="store_true", default=False)
    parser.add_argument("--intrinsic", type=str, default=None, choices=["none", "rnd", "counts"],
                        help="Override the intrinsic-reward mode (default: use the agent config as written).")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--no_visualise", action="store_true", default=False)
    parser.add_argument("--record", action="store_true", default=False,
                        help="Save visualiser snapshots to <log_dir>/snapshots for replay later.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    env_cls, cfg_cls, agents = TASKS[args.task]
    env_cfg = dataclasses.replace(cfg_cls(), **build_env_overrides(args.task, args))
    env = env_cls(env_cfg, device=args.device)

    if args.agent == "ppo":
        agent_cfg = agents.PPO_CFG
    else:
        agent_cfg = agents.SAC_CFG

        if agent_cfg.her_cfg is not None:
            her_cfg = dataclasses.replace(agent_cfg.her_cfg, n=env.dim, goal_radius=env_cfg.goal_radius)
            agent_cfg = dataclasses.replace(agent_cfg, her_cfg=her_cfg)

        agent_cfg = apply_her_overrides(agent_cfg, args)

    agent_cfg = apply_intrinsic_overrides(agent_cfg, args.agent, args.intrinsic, agents)
    print(agent_cfg)

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name is not None:
        log_dir = os.path.join(args.log_dir, args.task, run_name)
    else:
        log_dir = os.path.join(args.log_dir, args.task, agent_cfg.experiment_name, run_name)

    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "cfg.json"), "w") as f:
        json.dump(
            {
                "task": args.task,
                "agent": dataclasses.asdict(agent_cfg),
                "env": dataclasses.asdict(env_cfg),
            },
            f,
            indent=2,
        )

    runner = RLRunner(env=env, cfg=agent_cfg, log_dir=log_dir)

    if not args.no_visualise:
        env._visualisers.extend(build_visualisers(args.task, args.agent, runner, env, args.record))
    elif args.record:
        print("[WARN] --record has no effect with --no_visualise (no visualisers active).")

    try:
        runner.learn()
    finally:
        if args.record:
            snapshot_dir = os.path.join(log_dir, "snapshots")
            for v in env._visualisers:
                v.save(snapshot_dir)
            print(f"[INFO] Saved visualiser snapshots to {snapshot_dir}")
        writer = getattr(runner.runner, "writer", None)
        if writer is not None:
            writer.close()

    open(os.path.join(log_dir, "DONE"), "w").close()


if __name__ == "__main__":
    main()
