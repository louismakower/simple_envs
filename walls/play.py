"""Play a trained checkpoint on the walls maze: roll out a single env under the
deterministic policy and animate it live over the physical (x, y) maze.

Usage:
    python -m walls.play --checkpoint runs/walls/<run>/checkpoints/model_X.pth

The run's cfg.json (written by train.py next to the checkpoints/ dir) holds both
the agent and env configs under {"agent": ..., "env": ...}, so the maze layout
(walls, holes, seed) is rebuilt exactly rather than inferred.

The view is a single 2D plane of the unit square: the frozen wall/hole geometry
(cyan), the pinned goal (gold star, always at x=1), and the agent as a moving
dot trailing the path it has taken. The env auto-resets on success/timeout; on
each reset the trail is cleared and the new goal drawn. Loops until the window
is closed.

Pass --video N to record headlessly until N episodes have ended, write the clip
(mp4 via ffmpeg, else gif) to <run_dir>/rollout.mp4, and exit.
"""

import argparse
import dataclasses
import json
import os
import tempfile

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter

from louis_rl.algorithm import RLRunner
from louis_rl.intrinsic import RNDCfg
from louis_rl.ppo import PPORunnerCfg
from louis_rl.sac import SACRunnerCfg
from walls.env import WallsVecEnv
from walls.env_cfg import WallsVecEnvCfg
from walls.visualise import _draw_walls


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
    task = data.get("task")
    if task is not None and task != "walls":
        raise SystemExit(
            f"{cfg_path} is for task {task!r}, not 'walls'. "
            "Use the matching play script (e.g. python -m n_dim.play)."
        )
    return data["agent"], data["env"]


def _rebuild_rnd_cfg(data: dict) -> RNDCfg:
    """Rebuild an RNDCfg from an asdict() dump, tolerating schema drift.

    Older checkpoints predate fields later added to RNDCfg, so a plain
    ``RNDCfg(**data)`` raises on the missing arg. We keep the keys the current
    RNDCfg still has and fill any gap with that field's default — or a neutral
    placeholder for a since-added required field. This is safe because RND is
    never queried under the deterministic policy, so its config only needs to
    construct.
    """
    kwargs = {}
    for f in dataclasses.fields(RNDCfg):
        if f.name in data:
            kwargs[f.name] = data[f.name]
        elif f.default is not dataclasses.MISSING:
            kwargs[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            kwargs[f.name] = f.default_factory()
        else:
            kwargs[f.name] = 0  # unused at play time
    return RNDCfg(**kwargs)


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
        # asdict() flattened the nested rnd_cfg to a plain dict; rebuild the
        # dataclass since SACRunner constructs RND from it (even when disabled).
        if isinstance(agent_d.get("rnd_cfg"), dict):
            agent_d["rnd_cfg"] = _rebuild_rnd_cfg(agent_d["rnd_cfg"])
        agent_cfg = SACRunnerCfg(**agent_d)
    elif algo == "ppo":
        agent_cfg = PPORunnerCfg(**agent_d)
    else:
        raise SystemExit(f"Unknown algo_name {algo!r} in cfg.json")

    # Throwaway log_dir so the runner's SummaryWriter / checkpoints dir don't
    # land in the run we're replaying.
    log_dir = tempfile.mkdtemp(prefix="walls_play_")
    return RLRunner(env, agent_cfg, log_dir)


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


class WallsPanel:
    """The 2D maze with the agent dot, its trail, and the pinned goal.

    The frozen wall/hole geometry is drawn once from the env; the goal star and
    its acceptance circle move on each reset, and the trail grows each step.
    """

    def __init__(self, ax, env, goal_radius: float):
        self.ax = ax
        _draw_walls(
            ax,
            env.wall_x.cpu().numpy(),
            env.hole_lo.cpu().numpy(),
            env.hole_hi.cpu().numpy(),
        )
        (self.trail,) = ax.plot([], [], color="tab:blue", lw=1.0, alpha=0.6)
        (self.agent,) = ax.plot([], [], "o", color="tab:blue", ms=8)
        (self.goalmark,) = ax.plot(
            [], [], "*", color="gold", ms=16,
            markeredgecolor="black", markeredgewidth=0.6, zorder=6,
        )
        self.circle = plt.Circle((0, 0), goal_radius, fill=False, ls="--",
                                  color="tab:red", alpha=0.4)
        ax.add_patch(self.circle)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    def reset(self, goal):
        gx, gy = float(goal[0]), float(goal[1])
        self.goalmark.set_data([gx], [gy])
        self.circle.center = (gx, gy)
        self.trail.set_data([], [])
        self.agent.set_data([], [])

    def update(self, traj):
        xs = traj[:, 0]
        ys = traj[:, 1]
        self.trail.set_data(xs, ys)
        self.agent.set_data([xs[-1]], [ys[-1]])


# ---------------------------------------------------------------------------
# Rollout loop
# ---------------------------------------------------------------------------


def run_rollout(runner, env, panel, fig, *, delay, max_episodes=None,
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
        panel.update(np.asarray(traj))
        fig.suptitle(f"{title_prefix}episode {episode} — step {step}")
        fig.canvas.draw_idle()

    obs, _ = env.reset()
    panel.reset(goal_np())
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
            panel.reset(goal_np())
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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--delay", type=float, default=0.03,
                        help="Seconds to pause between animation frames.")
    parser.add_argument("--video", type=int, default=None, metavar="N",
                        help="Record until N episodes have ended, save the video, then exit.")
    parser.add_argument("--video_path", type=str, default=None,
                        help="Output path for --video (default: <run_dir>/rollout.mp4).")
    args = parser.parse_args()

    if args.video is not None and args.video < 1:
        raise SystemExit("--video must be a positive integer (number of episodes to record).")

    agent_d, env_d = load_cfgs(args.checkpoint)
    env_cfg = dataclasses.replace(WallsVecEnvCfg(**env_d), num_envs=1)

    if args.video is not None:
        matplotlib.use("Agg", force=True)  # headless: no display needed to record

    env = WallsVecEnv(env_cfg, device=args.device)
    runner = build_runner(agent_d, env)
    runner.load_checkpoint(args.checkpoint)
    print(f"[INFO] Loaded {args.checkpoint} "
          f"(n_walls={env_cfg.n_walls}, seed={env_cfg.seed}, algo={agent_d.get('algo_name')})")

    if args.video is None:
        plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    panel = WallsPanel(ax, env, env_cfg.goal_radius)
    fig.tight_layout()

    if args.video is not None:
        fps = max(1, round(1.0 / args.delay)) if args.delay > 0 else 30
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
        out_path = args.video_path or os.path.join(run_dir, "rollout.mp4")
        writer, out_path = make_writer(out_path, fps)
        out_dir = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(out_dir, exist_ok=True)
        hold = max(1, round(fps * 0.5))  # ~0.5s linger on each reached frame
        with writer.saving(fig, out_path, dpi=100):
            completed = run_rollout(runner, env, panel, fig, delay=args.delay,
                                    max_episodes=args.video, on_frame=writer.grab_frame,
                                    reset_hold_frames=hold)
        print(f"[INFO] Saved {completed} episode(s) to {out_path}")
    else:
        fig.show()
        run_rollout(runner, env, panel, fig, delay=args.delay)


if __name__ == "__main__":
    main()
