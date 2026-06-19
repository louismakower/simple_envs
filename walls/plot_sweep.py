"""Plot the walls sweep: one panel per n_walls, six curves each (2 algos x 3
intrinsic modes), mean +/- 1 std over seeds.

    python -m walls.plot_sweep

Reads the per-run TensorBoard scalars written under runs/walls_sweep, reusing
the fast event reader from n_dim.plot_sweep. Colour encodes the intrinsic mode,
line style the algorithm (solid PPO, dashed SAC). Saves reward_curves.png next
to the runs and shows it.
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from n_dim.plot_sweep import read_scalar
from walls.sweep import conditions, n_walls_values, seeds, log_path, LOG_DIR

INTRINSIC_COLORS = {"none": "tab:gray", "rnd": "tab:blue", "counts": "tab:orange"}
AGENT_STYLES = {"ppo": "-", "sac": "--"}


def load_all():
    jobs = [(cond, w, seed) for cond in conditions for w in n_walls_values for seed in seeds]

    def _load(job):
        cond, w, seed = job
        path = log_path(cond, w, seed)
        if not os.path.exists(path):
            return job, None
        return job, read_scalar(path, cond["reward_tag"])

    results = {}
    with ThreadPoolExecutor() as pool:
        for job, data in pool.map(_load, jobs):
            if data is not None:
                results[(job[0]["label"], job[1], job[2])] = data
    return results


def plot_results():
    data = load_all()

    ncols = math.ceil(math.sqrt(len(n_walls_values)))
    nrows = math.ceil(len(n_walls_values) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, w in zip(axes_flat, n_walls_values):
        for cond in conditions:
            seed_curves = []
            ref_steps = None
            for seed in seeds:
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

            curves = np.stack(seed_curves)
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            color = INTRINSIC_COLORS[cond["intrinsic"]]
            ls = AGENT_STYLES[cond["agent"]]

            ax.plot(ref_steps / 1e6, mean, color=color, linestyle=ls)
            ax.fill_between(ref_steps / 1e6, mean - std, mean + std, alpha=0.15, color=color)

        ax.set_title(f"n_walls={w}")
        ax.set_xlabel("Environment steps (M)")

    for ax in axes_flat[len(n_walls_values):]:
        ax.set_visible(False)

    axes_flat[0].set_ylabel("Mean extrinsic reward")

    legend_handles = [
        mlines.Line2D([], [], color=INTRINSIC_COLORS[c["intrinsic"]],
                      linestyle=AGENT_STYLES[c["agent"]], label=c["label"])
        for c in conditions
    ]
    fig.legend(handles=legend_handles, loc="center right", title="algo / intrinsic")
    fig.suptitle(f"Walls: extrinsic reward vs n_walls (±1 std, {len(seeds)} seeds)")
    fig.tight_layout(rect=[0, 0, 0.88, 1])

    out_path = os.path.join(LOG_DIR, "reward_curves.png")
    os.makedirs(LOG_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[INFO] saved {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_results()
