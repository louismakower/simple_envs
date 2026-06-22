"""Plot the walls sweep: one panel per n_walls, three curves each (one per
intrinsic mode) for a single algorithm, mean +/- 1 std over seeds.

    python -m walls.plot_sweep ppo
    python -m walls.plot_sweep sac

Reads the per-run TensorBoard scalars written under runs/walls_sweep, reusing
the fast event reader from n_dim.plot_sweep. Colour encodes the intrinsic mode.
Saves reward_curves_<agent>.png next to the runs and shows it.
"""

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from n_dim.plot_sweep import read_scalar
from walls.sweep import conditions, n_walls_values, seeds, log_path, LOG_DIR

INTRINSIC_COLORS = {"none": "tab:gray", "rnd": "tab:blue", "counts": "tab:orange"}
AGENT_STYLES = {"ppo": "-", "sac": "--"}  # used to distinguish algos when plotting both
SMOOTH_WINDOW = 500  # rolling mean over this many logged points (1 point / 256 env steps)


def rolling_mean(y, window):
    """Centred rolling mean over `window` points, normalised at the edges so the
    curve isn't pulled towards zero at the start/end."""
    if window <= 1 or len(y) < 2:
        return y
    kernel = np.ones(min(window, len(y)))
    counts = np.convolve(np.ones_like(y, dtype=float), kernel, mode="same")
    return np.convolve(y, kernel, mode="same") / counts


def _load(job):
    cond, w, seed = job
    path = log_path(cond, w, seed)
    if not os.path.exists(path):
        return job, None
    return job, read_scalar(path, cond["reward_tag"])


def load_all(conditions):
    jobs = [(cond, w, seed) for cond in conditions for w in n_walls_values for seed in seeds]

    # Parsing the (large, ~200 MB) SAC event files is CPU-bound and pure-Python,
    # so threads serialise on the GIL. Use processes for real parallelism; once
    # read_scalar has written its .npz caches, subsequent runs are cache hits.
    results = {}
    with ProcessPoolExecutor() as pool:
        for job, data in pool.map(_load, jobs):
            print("running", job)
            if data is not None:
                results[(job[0]["label"], job[1], job[2])] = data
    return results


def plot_results(agent, smooth=SMOOTH_WINDOW):
    sel = conditions if agent == "both" else [c for c in conditions if c["agent"] == agent]
    show_styles = len({c["agent"] for c in sel}) > 1  # distinguish algos by line style
    data = load_all(sel)

    ncols = math.ceil(math.sqrt(len(n_walls_values)))
    nrows = math.ceil(len(n_walls_values) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, w in zip(axes_flat, n_walls_values):
        for cond in sel:
            seed_curves = []
            ref_steps = None
            for seed in seeds:
                print("getting")
                entry = data.get((cond["label"], w, seed))
                if entry is None:
                    continue
                steps, values = entry
                if len(steps) == 0:
                    continue
                if ref_steps is None:
                    ref_steps = steps
                if len(steps) == len(ref_steps):
                    seed_curves.append(values)
                else:
                    seed_curves.append(np.interp(ref_steps, steps, values))

            if not seed_curves:
                continue

            curves = np.stack([rolling_mean(c, smooth) for c in seed_curves])
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            color = INTRINSIC_COLORS[cond["intrinsic"]]
            ls = AGENT_STYLES[cond["agent"]] if show_styles else "-"

            ax.plot(ref_steps / 1e6, mean, color=color, linestyle=ls)
            ax.fill_between(ref_steps / 1e6, mean - std, mean + std, alpha=0.15, color=color)

        ax.set_title(f"n_walls={w}")
        ax.set_xlabel("Environment steps (M)")

    for ax in axes_flat[len(n_walls_values):]:
        ax.set_visible(False)

    axes_flat[0].set_ylabel("Mean extrinsic reward")

    if show_styles:
        legend_handles = [
            mlines.Line2D([], [], color=INTRINSIC_COLORS[c["intrinsic"]],
                          linestyle=AGENT_STYLES[c["agent"]], label=c["label"])
            for c in sel
        ]
        legend_title = "algo / intrinsic"
    else:
        legend_handles = [
            mlines.Line2D([], [], color=INTRINSIC_COLORS[c["intrinsic"]], label=c["intrinsic"])
            for c in sel
        ]
        legend_title = "intrinsic"
    fig.legend(handles=legend_handles, loc="center right", title=legend_title)
    fig.suptitle(f"Walls ({agent.upper()}): extrinsic reward vs n_walls (±1 std, {len(seeds)} seeds)")
    fig.tight_layout(rect=[0, 0, 0.88, 1])

    out_path = os.path.join(LOG_DIR, f"reward_curves_{agent}.png")
    os.makedirs(LOG_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[INFO] saved {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the walls sweep for one or both algorithms.")
    parser.add_argument("agent", choices=["ppo", "sac", "both"], help="algorithm(s) to plot")
    parser.add_argument("--smooth", type=int, default=SMOOTH_WINDOW,
                        help="rolling-mean window in logged points (1 point = 256 env steps)")
    args = parser.parse_args()
    plot_results(args.agent, smooth=args.smooth)
