"""Replay recorded visualiser snapshots from a training run.

Usage:
    python -m n_dim.replay <run_dir>

Reads every `.npz` under `<run_dir>/snapshots/` and opens one matplotlib
window per visualiser. A separate controller window shows the reward curve
and a single shared slider whose position drives all the visualisers, plus
a marker on the reward curve at the current step.
"""

import argparse
import math
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from n_dim.plot_sweep import read_scalar


# ---------------------------------------------------------------------------
# Per-visualiser renderers
#
# Each returns (fig, on_step(step_value) -> None, (min_step, max_step)).
# The on_step callback is responsible for snapping to the nearest recorded
# snapshot at-or-before `step_value` and redrawing the figure.
# ---------------------------------------------------------------------------


def _snap_index(steps, target_step):
    """Largest idx with steps[idx] <= target_step, clamped to [0, len-1]."""
    idx = int(np.searchsorted(steps, target_step, side="right") - 1)
    return max(0, min(idx, len(steps) - 1))


def render_policy(data):
    n = int(data["n"])
    res = int(data["quiver_res"])
    goals = data["goals"]
    uvs = data["uv"]
    steps = data["steps"]

    if n == 1:
        goal_labels = [float(g) for g in goals]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    else:
        goal_labels = [tuple(g.tolist()) for g in goals]
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = list(np.atleast_1d(axes).ravel())
    try:
        fig.canvas.manager.set_window_title("Policy (replay)")
    except Exception:
        pass

    axis_vals = np.linspace(0.0, 1.0, res)
    X, Y = np.meshgrid(axis_vals, axis_vals)
    quivers = []
    for ax, goal, uv in zip(axes, goal_labels, uvs[0]):
        q = ax.quiver(
            X, Y, uv[0], uv[1],
            angles="xy", scale_units="xy", scale=1.0, width=0.004,
        )
        quivers.append(q)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.set_title(f"goal = {goal}")
        ax.set_xlabel("state dim 0")
        ax.set_ylabel("state dim 1" if n >= 2 else "(padded)")

    title = fig.suptitle(f"Policy (replay) — step {int(steps[0])}")
    fig.tight_layout()

    last_idx = [0]

    def on_step(step_value):
        t = _snap_index(steps, step_value)
        if t == last_idx[0]:
            return
        last_idx[0] = t
        for k, q in enumerate(quivers):
            uv = uvs[t, k]
            q.set_UVC(uv[0], uv[1])
        title.set_text(f"Policy (replay) — step {int(steps[t])}")
        fig.canvas.draw_idle()

    return fig, on_step, (int(steps[0]), int(steps[-1]))


def render_value(data):
    title = str(data["title"]) if "title" in data.files else "Value vs distance to goal"
    ylabel = str(data["ylabel"]) if "ylabel" in data.files else "V"
    dists = data["dists"]
    values = data["values"]
    steps = data["steps"]
    n = int(data["n"])

    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        fig.canvas.manager.set_window_title(f"{title} (replay)")
    except Exception:
        pass
    scatter = ax.scatter(dists[0], values[0], s=4, alpha=0.4)
    ax.set_xlim(0.0, math.sqrt(n))
    ax.set_xlabel("distance to goal")
    ax.set_ylabel(ylabel)
    ax.set_ylim(float(values[0].min()), float(values[0].max()) + 1e-6)
    ax_title = ax.set_title(f"{title} — step {int(steps[0])}")
    fig.tight_layout()

    last_idx = [0]

    def on_step(step_value):
        t = _snap_index(steps, step_value)
        if t == last_idx[0]:
            return
        last_idx[0] = t
        scatter.set_offsets(np.column_stack([dists[t], values[t]]))
        ax.set_ylim(float(values[t].min()), float(values[t].max()) + 1e-6)
        ax_title.set_text(f"{title} — step {int(steps[t])}")
        fig.canvas.draw_idle()

    return fig, on_step, (int(steps[0]), int(steps[-1]))


def render_buffer(data):
    mean_reward = data["mean_reward"]
    counts = data["counts"]
    steps = data["steps"]
    n = int(data["n"])

    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n), squeeze=False)
    try:
        fig.canvas.manager.set_window_title("Buffer (replay)")
    except Exception:
        pass
    rew_vmin = -0.01
    rew_vmax = 10 * n - 0.01

    reward_images = []
    count_images = []
    for k in range(n):
        rim = axes[k, 0].imshow(
            mean_reward[0, k].T,
            origin="lower", extent=[0, 1, 0, 1],
            aspect="equal", interpolation="nearest",
            cmap="RdYlGn", vmin=rew_vmin, vmax=rew_vmax,
        )
        fig.colorbar(rim, ax=axes[k, 0], label="mean reward")
        axes[k, 0].set_xlabel(f"state dim {k}")
        axes[k, 0].set_ylabel(f"goal dim {k}")
        axes[k, 0].set_title(f"dim {k} — mean reward")
        reward_images.append(rim)

        cnt = counts[0, k]
        cnt_grid = np.where(cnt > 0, cnt, np.nan)
        cim = axes[k, 1].imshow(
            cnt_grid.T,
            origin="lower", extent=[0, 1, 0, 1],
            aspect="equal", interpolation="nearest",
            cmap="Blues",
        )
        fig.colorbar(cim, ax=axes[k, 1], label="transitions")
        axes[k, 1].set_xlabel(f"state dim {k}")
        axes[k, 1].set_ylabel(f"goal dim {k}")
        axes[k, 1].set_title(f"dim {k} — transitions")
        count_images.append(cim)

    title = fig.suptitle(f"Replay buffer — step {int(steps[0])}")
    fig.tight_layout()

    last_idx = [0]

    def on_step(step_value):
        t = _snap_index(steps, step_value)
        if t == last_idx[0]:
            return
        last_idx[0] = t
        for k in range(n):
            mr = mean_reward[t, k]
            cnt = counts[t, k]
            reward_images[k].set_data(mr.T)
            cnt_grid = np.where(cnt > 0, cnt, np.nan)
            count_images[k].set_data(cnt_grid.T)
            cmax = cnt.max() if cnt.max() > 0 else 1.0
            count_images[k].set_clim(vmin=0, vmax=cmax)
        title.set_text(f"Replay buffer — step {int(steps[t])}")
        fig.canvas.draw_idle()

    return fig, on_step, (int(steps[0]), int(steps[-1]))


DISPATCH = {
    "policy": render_policy,
    "value": render_value,
    "sac_value": render_value,
    "ppo_value": render_value,
    "buffer": render_buffer,
}


# ---------------------------------------------------------------------------
# Reward curve loader
# ---------------------------------------------------------------------------


REWARD_TAG_CANDIDATES = ("rewards/extrinsic_mean", "rewards/mean")


def _rolling_mean(x, window):
    """Centred rolling mean; edges divide by actual valid count so they aren't
    diluted toward zero."""
    if window <= 1 or len(x) == 0:
        return x.astype(np.float64, copy=True)
    kernel = np.ones(window, dtype=np.float64)
    sums = np.convolve(x.astype(np.float64), kernel, mode="same")
    counts = np.convolve(np.ones(len(x), dtype=np.float64), kernel, mode="same")
    return sums / counts


def load_reward_curve(run_dir):
    """Return (steps, values) in env-step units, or (None, None) if unavailable.

    The tensorboard step is the runner's iteration counter; the visualiser
    step is the env-step counter. We don't read steps_per_rollout — instead
    the caller scales the tb-step axis to match the visualiser step range.
    """
    for tag in REWARD_TAG_CANDIDATES:
        steps, values = read_scalar(run_dir, tag)
        if len(steps) > 0:
            return np.asarray(steps), np.asarray(values), tag
    return None, None, None


# ---------------------------------------------------------------------------
# Controller: reward curve + marker + shared slider
# ---------------------------------------------------------------------------


def make_controller(run_dir, slider_min, slider_max, callbacks, smooth_window=300):
    reward_steps, reward_values, reward_tag = load_reward_curve(run_dir)

    fig, ax = plt.subplots(figsize=(9, 4))
    try:
        fig.canvas.manager.set_window_title("Controller (replay)")
    except Exception:
        pass

    marker = None
    reward_x = None
    reward_y_smoothed = None
    if reward_steps is not None and len(reward_steps) >= 2:
        # Scale tb-step axis so its max aligns with the visualiser step max.
        # This puts both signals on the same x-axis without needing
        # steps_per_rollout from the config.
        tb_max = float(reward_steps[-1])
        scale = slider_max / tb_max if tb_max > 0 else 1.0
        reward_x = reward_steps.astype(np.float64) * scale

        # Convert the env-step smoothing window to a sample count using the
        # median spacing between reward points on the scaled axis.
        median_dx = float(np.median(np.diff(reward_x))) if len(reward_x) >= 2 else 1.0
        window_samples = max(1, int(round(smooth_window / median_dx))) if median_dx > 0 else 1
        reward_y_smoothed = _rolling_mean(reward_values, window_samples)

        ax.plot(reward_x, reward_values, color="tab:blue", linewidth=0.8, alpha=0.25, label="raw")
        ax.plot(reward_x, reward_y_smoothed, color="tab:blue", linewidth=1.6,
                label=f"rolling mean ({smooth_window} env steps, ~{window_samples} samples)")
        marker, = ax.plot(
            [reward_x[0]], [reward_y_smoothed[0]],
            marker="o", color="tab:red", markersize=8, zorder=5,
        )
        ax.set_ylabel(reward_tag)
        ax.set_title(f"Reward curve — {reward_tag}")
        ax.legend(loc="lower right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No reward curve found in run dir",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Reward curve (unavailable)")
        ax.set_yticks([])

    ax.set_xlabel("env step")
    ax.set_xlim(slider_min, slider_max)
    fig.subplots_adjust(bottom=0.28)
    slider_ax = fig.add_axes([0.12, 0.10, 0.78, 0.04])
    slider = Slider(
        slider_ax, "step",
        slider_min, slider_max,
        valinit=slider_min, valstep=1,
    )

    def on_change(_val):
        s = float(slider.val)
        for cb in callbacks:
            cb(s)
        if marker is not None and reward_x is not None:
            idx = int(np.searchsorted(reward_x, s))
            idx = max(0, min(idx, len(reward_x) - 1))
            marker.set_data([reward_x[idx]], [reward_y_smoothed[idx]])
        fig.canvas.draw_idle()

    slider.on_changed(on_change)
    fig._replay_slider = slider  # keep reference alive
    return fig


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str,
                        help="Run directory containing a snapshots/ folder")
    parser.add_argument("--smooth_window", type=int, default=300,
                        help="Rolling-mean window for the reward curve, in env steps.")
    args = parser.parse_args()

    snapshot_dir = os.path.join(args.run_dir, "snapshots")
    if not os.path.isdir(snapshot_dir):
        raise SystemExit(f"No snapshots/ directory found in {args.run_dir}")

    callbacks = []
    step_ranges = []
    for fname in sorted(os.listdir(snapshot_dir)):
        if not fname.endswith(".npz"):
            continue
        path = os.path.join(snapshot_dir, fname)
        data = np.load(path)
        kind = str(data["kind"])
        renderer = DISPATCH.get(kind)
        if renderer is None:
            print(f"[WARN] unknown kind {kind!r} in {fname}, skipping")
            continue
        if data["steps"].size == 0:
            print(f"[WARN] no frames recorded in {fname}, skipping")
            continue
        _fig, on_step, (lo, hi) = renderer(data)
        callbacks.append(on_step)
        step_ranges.append((lo, hi))
        print(f"[INFO] loaded {fname}: {data['steps'].size} frames "
              f"({kind}, steps {lo}..{hi})")

    if not callbacks:
        raise SystemExit(f"No snapshots loaded from {snapshot_dir}")

    slider_min = min(lo for lo, _ in step_ranges)
    slider_max = max(hi for _, hi in step_ranges)
    make_controller(args.run_dir, slider_min, slider_max, callbacks,
                    smooth_window=args.smooth_window)

    plt.show()


if __name__ == "__main__":
    main()
