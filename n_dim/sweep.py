import os
import subprocess
import sys
from itertools import product

conditions = [
    dict(label="PPO",     agent="ppo", extra_args=[]),
    dict(label="SAC+HER", agent="sac", extra_args=[]),
    dict(label="SAC",     agent="sac", extra_args=["--disable_her"]),
]

n_values = [5,7,9]
num_envs = 1024
seeds = [42, 43, 44]

LOG_DIR = "runs/sweep"
TASK = "n_dim"


def run_name(cond, n, seed):
    return f"{cond['label'].lower().replace('+', '_')}/n{n}_seed{seed}"


def log_path(cond, n, seed):
    # train.py writes to <log_dir>/<task>/<run_name>; mirror that here.
    return os.path.join(LOG_DIR, TASK, run_name(cond, n, seed))


def run_training():
    for cond, n, seed in product(conditions, n_values, seeds):
        path = log_path(cond, n, seed)
        if os.path.exists(os.path.join(path, "DONE")):
            print(f"[skip] {cond['label']} n={n} seed={seed}")
            continue
        print(f"[run]  {cond['label']} n={n} seed={seed}")
        cmd = [
            sys.executable, "-m", "train",
            "--task", TASK,
            "--agent", cond["agent"],
            "--num_envs", str(num_envs),
            "--n", str(n),
            "--seed", str(seed),
            "--run_name", run_name(cond, n, seed),
            "--log_dir", LOG_DIR,
            "--no_visualise",
            "--device", "cuda",
            *cond["extra_args"],
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run_training()
