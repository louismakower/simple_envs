"""Plot reward curves for the HER sweep produced by run_her_sweep.py.

For each environment it reads the `rewards/extrinsic_mean` TensorBoard scalar
from every sweep run, groups the 5 seeds per HER config, aligns them onto a
common step grid and plots the mean with a +/- std band -- one line per config.

Outputs:
  runs/her_sweep/sac_1d_reward_curves.png
  runs/her_sweep/sac_2d_reward_curves.png

Usage:
  python plot_her_sweep.py                 # plot both envs
  python plot_her_sweep.py --smooth 50     # heavier smoothing window
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_ROOT = os.path.join(ROOT, "runs", "her_sweep")
EXPS = {"one_dim": "sac_1d", "two_dim": "sac_2d"}
SCALAR = "rewards/extrinsic_mean"
GRID_POINTS = 400

# Fixed config order so colours/legend are stable across the two env plots.
CONFIG_ORDER = ["nohier",
                "future_k1", "future_k2", "future_k4",
                "final_k1", "final_k2", "final_k4"]


def read_scalar(run_dir):
    """Return (steps, values) arrays for SCALAR in run_dir, or None if absent."""
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if SCALAR not in ea.Tags().get("scalars", []):
        return None
    events = ea.Scalars(SCALAR)
    steps = np.array([e.step for e in events], dtype=float)
    values = np.array([e.value for e in events], dtype=float)
    return steps, values


def smooth(values, window):
    """Centred rolling mean; window <= 1 is a no-op."""
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def collect_runs(exp_dir):
    """Map config_name -> list of (steps, values) over its seeds."""
    runs = {}
    if not os.path.isdir(exp_dir):
        return runs
    for name in sorted(os.listdir(exp_dir)):
        run_dir = os.path.join(exp_dir, name)
        if not os.path.isdir(run_dir) or "_seed" not in name:
            continue
        config_name = name.rsplit("_seed", 1)[0]
        curve = read_scalar(run_dir)
        if curve is None:
            print(f"[WARN] no '{SCALAR}' scalar in {run_dir}")
            continue
        runs.setdefault(config_name, []).append(curve)
    return runs


def aggregate(curves, smooth_window):
    """Align seed curves onto a common grid; return (grid, mean, std)."""
    first = max(steps.min() for steps, _ in curves)
    last = min(steps.max() for steps, _ in curves)
    grid = np.linspace(first, last, GRID_POINTS)
    aligned = []
    for steps, values in curves:
        order = np.argsort(steps)
        interp = np.interp(grid, steps[order], values[order])
        aligned.append(smooth(interp, smooth_window))
    stacked = np.vstack(aligned)
    return grid, stacked.mean(axis=0), stacked.std(axis=0)


def plot_env(env, exp, smooth_window):
    exp_dir = os.path.join(SWEEP_ROOT, exp)
    runs = collect_runs(exp_dir)
    if not runs:
        print(f"[SKIP] no runs found under {exp_dir}")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ordered = [c for c in CONFIG_ORDER if c in runs]
    ordered += [c for c in sorted(runs) if c not in CONFIG_ORDER]
    cmap = plt.get_cmap("tab10")

    for idx, config_name in enumerate(ordered):
        curves = runs[config_name]
        grid, mean, std = aggregate(curves, smooth_window)
        colour = cmap(idx % 10)
        label = f"{config_name} (n={len(curves)})"
        ax.plot(grid, mean, label=label, color=colour, linewidth=2)
        ax.fill_between(grid, mean - std, mean + std, color=colour, alpha=0.15)

    ax.set_xlabel("training step")
    ax.set_ylabel(f"{SCALAR} (mean +/- std over seeds)")
    ax.set_title(f"HER sweep -- {env} ({exp})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(SWEEP_ROOT, f"{exp}_reward_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--envs", nargs="+", choices=list(EXPS), default=list(EXPS),
                        help="Which environments to plot (default: both).")
    parser.add_argument("--smooth", type=int, default=20,
                        help="Rolling-mean window applied per seed before aggregating.")
    args = parser.parse_args()

    for env in args.envs:
        plot_env(env, EXPS[env], args.smooth)


if __name__ == "__main__":
    main()
