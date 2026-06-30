"""Play a trained checkpoint: roll out a single env under the deterministic
policy and animate it live.

Usage:
    python -m n_dim.play --checkpoint runs/sac_nd/<run>/checkpoints/model_X.pth --mode 2d

The run's cfg.json (written by train.py next to the checkpoints/ dir) holds both
the agent and env configs under {"agent": ..., "env": ...}, so n and the env
params are read directly rather than inferred.

Two view modes:
  1d  -> N subplots, one per state dim, showing that dim's position vs time
         with the goal as a horizontal line.
  2d  -> ceil(N/2) subplots, each a 2D plane (dim 2k, dim 2k+1) with the agent
         as a moving dot, a trail of the path, and the goal as a star. An odd
         leftover dim falls back to a 1d position-vs-time subplot.

The env auto-resets on success/timeout; on each reset the trail / time axis is
cleared and the new goal drawn. Loops until the window is closed.

Pass --video N to record headlessly until N episodes have ended, write the
clip (mp4 via ffmpeg, else gif) to <run_dir>/rollout_<mode>.mp4, and exit.
"""

import argparse
import dataclasses
import json
import math
import os
import tempfile

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter

from louis_rl.rl_runner import RLRunner
from louis_rl.algos.ppo import PPORunnerCfg
from louis_rl.algos.sac import SACRunnerCfg
from n_dim.env import NDimVecEnv
from n_dim.env_cfg import NDimVecEnvCfg


# ---------------------------------------------------------------------------
# Config / runner loading
# ---------------------------------------------------------------------------


def load_cfgs(checkpoint: str):
    """Return (agent_dict, env_dict) read from <run_dir>/cfg.json.

    The checkpoint lives at <run_dir>/checkpoints/model_X.pth, so the run dir
    is two levels up.
    """
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint)))
    cfg_path = os.path.join(run_dir, "cfg.json")
    if not os.path.isfile(cfg_path):
        raise SystemExit(f"No cfg.json found at {cfg_path}")
    with open(cfg_path) as f:
        data = json.load(f)
    if "agent" not in data or "env" not in data:
        raise SystemExit(
            f"{cfg_path} predates env-cfg logging; expected top-level "
            "'agent' and 'env' keys. Retrain with the updated train.py."
        )
    return data["agent"], data["env"]


def build_runner(agent_d: dict, env):
    """Rebuild the runner from a saved agent-cfg dict and load nothing yet.

    For SAC we drop the HER config and shrink the replay buffer to 1: neither
    affects the deterministic policy, but the default 10M buffer would allocate
    a large tensor on the device just to play.
    """
    algo = str(agent_d.get("algo_name", "")).lower()
    agent_d = dict(agent_d)
    if algo == "sac":
        agent_d["her_cfg"] = None
        agent_d["replay_buffer_size"] = 1
        agent_cfg = SACRunnerCfg(**agent_d)
    elif algo == "ppo":
        agent_cfg = PPORunnerCfg(**agent_d)
    else:
        raise SystemExit(f"Unknown algo_name {algo!r} in cfg.json")

    # Throwaway log_dir so the runner's SummaryWriter / checkpoints dir don't
    # land in the run we're replaying.
    log_dir = tempfile.mkdtemp(prefix="nd_play_")
    return RLRunner(env, agent_cfg, log_dir)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


class TimePanel:
    """One state dimension as position-vs-time."""

    def __init__(self, ax, k: int, n: int, max_ep_len: int, goal_radius: float):
        self.ax = ax
        self.k = k
        self.r = goal_radius
        self.show_band = (n == 1)  # per-dim band is only exact in 1D
        self.band = None

        (self.line,) = ax.plot([], [], color="tab:blue", lw=1.5)
        self.goal_line = ax.axhline(0.0, color="tab:red", ls="--", lw=1.2)
        ax.set_xlim(0, max_ep_len)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("step")
        ax.set_ylabel(f"state dim {k}")
        ax.set_title(f"dim {k}")

    def reset(self, goal):
        g = float(goal[self.k])
        self.goal_line.set_ydata([g, g])
        if self.show_band:
            if self.band is not None:
                self.band.remove()
            self.band = self.ax.axhspan(g - self.r, g + self.r, color="tab:red", alpha=0.12)
        self.line.set_data([], [])

    def update(self, traj):
        ys = traj[:, self.k]
        self.line.set_data(np.arange(len(ys)), ys)


class PlanePanel:
    """A pair of state dimensions as a 2D plane with a moving dot + trail."""

    def __init__(self, ax, a: int, b: int, n: int, goal_radius: float):
        self.ax = ax
        self.a = a
        self.b = b
        (self.trail,) = ax.plot([], [], color="tab:blue", lw=1.0, alpha=0.6)
        (self.agent,) = ax.plot([], [], "o", color="tab:blue", ms=8)
        (self.goalmark,) = ax.plot([], [], "*", color="tab:red", ms=16)
        self.circle = None
        if n == 2:  # the plane is the whole space, so the radius is exact
            self.circle = plt.Circle((0, 0), goal_radius, fill=False,
                                     ls="--", color="tab:red", alpha=0.4)
            ax.add_patch(self.circle)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel(f"state dim {a}")
        ax.set_ylabel(f"state dim {b}")
        ax.set_title(f"dims {a}, {b}")

    def reset(self, goal):
        gx, gy = float(goal[self.a]), float(goal[self.b])
        self.goalmark.set_data([gx], [gy])
        if self.circle is not None:
            self.circle.center = (gx, gy)
        self.trail.set_data([], [])
        self.agent.set_data([], [])

    def update(self, traj):
        xs = traj[:, self.a]
        ys = traj[:, self.b]
        self.trail.set_data(xs, ys)
        self.agent.set_data([xs[-1]], [ys[-1]])


def make_panels(mode: str, n: int, max_ep_len: int, goal_radius: float):
    """Build the figure + list of panels for the chosen mode."""
    if mode == "1d":
        num = n
    else:  # 2d: planes for paired dims, plus a 1d panel for an odd leftover
        num = (n + 1) // 2

    ncols = math.ceil(math.sqrt(num))
    nrows = math.ceil(num / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows), squeeze=False)
    flat = list(axes.ravel())
    for ax in flat[num:]:
        ax.axis("off")

    panels = []
    if mode == "1d":
        for k in range(n):
            panels.append(TimePanel(flat[k], k, n, max_ep_len, goal_radius))
    else:
        idx = 0
        for k in range(n // 2):
            panels.append(PlanePanel(flat[idx], 2 * k, 2 * k + 1, n, goal_radius))
            idx += 1
        if n % 2 == 1:  # leftover dim shown as position-vs-time
            panels.append(TimePanel(flat[idx], n - 1, n, max_ep_len, goal_radius))

    return fig, panels


# ---------------------------------------------------------------------------
# Rollout loop
# ---------------------------------------------------------------------------


def run_rollout(runner, env, panels, fig, *, delay, max_episodes=None,
                on_frame=None, reset_hold_frames=1, title_prefix=""):
    """Drive a single env under the deterministic policy, redrawing each step.

    Stops once the window is closed, or (if max_episodes is set) once that many
    episodes have ended. When `on_frame` is given it is called after every
    redraw (to grab a video frame) and the loop runs flat out; otherwise it
    pauses in real time so the window is watchable. Returns the number of
    completed episodes.
    """
    recording = on_frame is not None

    def state_np():
        return env.state[0].detach().cpu().numpy().copy()

    def goal_np():
        return env.goal[0].detach().cpu().numpy().copy()

    def redraw(traj, episode, step):
        arr = np.asarray(traj)
        for p in panels:
            p.update(arr)
        fig.suptitle(f"{title_prefix}episode {episode} — step {step}")
        fig.canvas.draw_idle()

    obs, _ = env.reset()
    goal = goal_np()
    for p in panels:
        p.reset(goal)
    traj = [state_np()]
    episode = 1
    completed = 0
    reset_hold = max(delay, 0.4)  # brief pause on the frame the goal is reached

    while recording or plt.fignum_exists(fig.number):
        with torch.no_grad():
            action = runner.get_deterministic_action(obs)
        obs, _rew, term, timeout, extras = env.step(action)

        if bool((term | timeout)[0].item()):
            terminal = extras["terminal_obs"]["policy"][0].detach().cpu().numpy().copy()
            traj.append(terminal)
            redraw(traj, episode, len(traj) - 1)
            if recording:
                for _ in range(reset_hold_frames):  # linger on the reached frame
                    on_frame()
            else:
                plt.pause(reset_hold)
            completed += 1
            if max_episodes is not None and completed >= max_episodes:
                break
            # env already auto-reset inside step(): pick up the new goal/start.
            goal = goal_np()
            for p in panels:
                p.reset(goal)
            traj = [state_np()]
            episode += 1
            continue

        traj.append(state_np())
        redraw(traj, episode, len(traj) - 1)
        if recording:
            on_frame()
        else:
            plt.pause(delay)

    return completed


def make_writer(path: str, fps: int):
    """Pick a video writer from the output extension. Returns (writer, path).

    Defaults to mp4 via ffmpeg; uses Pillow for .gif, and falls back to gif if
    ffmpeg isn't available.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gif":
        return PillowWriter(fps=fps), path
    if FFMpegWriter.isAvailable():
        return FFMpegWriter(fps=fps), path
    alt = os.path.splitext(path)[0] + ".gif"
    print(f"[WARN] ffmpeg unavailable; saving GIF to {alt}")
    return PillowWriter(fps=fps), alt


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a model_X.pth checkpoint.")
    parser.add_argument("--mode", type=str, default="2d", choices=["1d", "2d"],
                        help="1d: N position-vs-time plots. 2d: N/2 plane plots.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--delay", type=float, default=0.03,
                        help="Seconds to pause between animation frames.")
    parser.add_argument("--video", type=int, default=None, metavar="N",
                        help="Record until N episodes have ended, save the video, then exit.")
    parser.add_argument("--video_path", type=str, default=None,
                        help="Output path for --video (default: <run_dir>/rollout_<mode>.mp4).")
    args = parser.parse_args()

    if args.video is not None and args.video < 1:
        raise SystemExit("--video must be a positive integer (number of episodes to record).")

    agent_d, env_d = load_cfgs(args.checkpoint)
    env_cfg = dataclasses.replace(NDimVecEnvCfg(**env_d), num_envs=1)

    if args.video is not None:
        matplotlib.use("Agg", force=True)  # headless: no display needed to record

    env = NDimVecEnv(env_cfg, device=args.device)
    runner = build_runner(agent_d, env)
    runner.load_checkpoint(args.checkpoint)
    print(f"[INFO] Loaded {args.checkpoint} (n={env_cfg.n}, algo={agent_d.get('algo_name')})")

    if args.video is None:
        plt.ion()
    fig, panels = make_panels(args.mode, env_cfg.n, env_cfg.max_ep_len, env_cfg.goal_radius)
    fig.tight_layout()

    if args.video is not None:
        fps = max(1, round(1.0 / args.delay)) if args.delay > 0 else 30
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
        out_path = args.video_path or os.path.join(run_dir, f"rollout_{args.mode}.mp4")
        writer, out_path = make_writer(out_path, fps)
        out_dir = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(out_dir, exist_ok=True)
        hold = max(1, round(fps * 0.5))  # ~0.5s linger on each reached frame
        with writer.saving(fig, out_path, dpi=100):
            completed = run_rollout(runner, env, panels, fig, delay=args.delay,
                                    max_episodes=args.video, on_frame=writer.grab_frame,
                                    reset_hold_frames=hold)
        print(f"[INFO] Saved {completed} episode(s) to {out_path}")
    else:
        fig.show()
        run_rollout(runner, env, panels, fig, delay=args.delay)


if __name__ == "__main__":
    main()
