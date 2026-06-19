"""Walls sweep: number of walls x algorithm x intrinsic-reward mode.

Drives train.py once per (condition, n_walls, seed) via subprocess, skipping any
run that already wrote a DONE file so the sweep is resumable. Mirrors
n_dim/sweep.py but adds the intrinsic axis (none | rnd | counts) and fixes the
maze layout per n_walls (train.py defaults maze_seed=42) so the only varying RNG
is the agent seed.

    python -m walls.sweep            # run/resume the whole sweep
    python -m walls.plot_sweep       # plot the reward curves afterwards

2 algos x 3 intrinsic modes x 8 wall counts x 3 seeds = 144 runs. SAC runs with
HER disabled so the only exploration axis that varies is the intrinsic reward.
"""

import os
import subprocess
import sys
from itertools import product

# extrinsic / task reward, logged by both algos regardless of intrinsic mode.
PPO_REWARD_TAG = "rewards/mean"
SAC_REWARD_TAG = "rewards/extrinsic_mean"

conditions = []
for _agent, _reward_tag, _base_args in [
    ("ppo", PPO_REWARD_TAG, []),
    ("sac", SAC_REWARD_TAG, ["--disable_her"]),
]:
    for _intrinsic in ["none", "rnd", "counts"]:
        conditions.append(dict(
            label=f"{_agent.upper()}/{_intrinsic}",
            agent=_agent,
            intrinsic=_intrinsic,
            reward_tag=_reward_tag,
            extra_args=_base_args + ["--intrinsic", _intrinsic],
        ))

n_walls_values = list(range(1, 9))  # 1..8
num_envs = 256
seeds = [42, 43, 44]

LOG_DIR = "runs/walls_sweep"
TASK = "walls"
DEVICE = "cuda"


def run_name(cond, w, seed):
    safe = cond["label"].lower().replace("/", "_")
    return f"{safe}/nwalls{w}_seed{seed}"


def log_path(cond, w, seed):
    # train.py writes to <log_dir>/<task>/<run_name>; mirror that here.
    return os.path.join(LOG_DIR, TASK, run_name(cond, w, seed))


def run_training():
    for cond, w, seed in product(conditions, n_walls_values, seeds):
        path = log_path(cond, w, seed)
        if os.path.exists(os.path.join(path, "DONE")):
            print(f"[skip] {cond['label']} n_walls={w} seed={seed}")
            continue
        print(f"[run]  {cond['label']} n_walls={w} seed={seed}")
        cmd = [
            sys.executable, "-m", "train",
            "--task", TASK,
            "--agent", cond["agent"],
            "--num_envs", str(num_envs),
            "--n_walls", str(w),
            "--seed", str(seed),
            "--run_name", run_name(cond, w, seed),
            "--log_dir", LOG_DIR,
            "--no_visualise",
            "--device", DEVICE,
            *cond["extra_args"],
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run_training()
