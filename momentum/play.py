"""Play a trained checkpoint on the momentum walls maze.

The momentum env is a `WallsVecEnv` (same frozen maze geometry, same pinned
goal, same 2D view) whose policy obs is `[position, velocity]`. Everything the
walls player does at play time — rebuild the runner from cfg.json, roll out one
env under the deterministic policy, animate / record over the maze — is env
agnostic, so we reuse those helpers wholesale and only swap in the momentum env.

Usage:
    python -m momentum.play --checkpoint runs/momentum/<run>/checkpoints/model_X.pth
    python -m momentum.play --checkpoint ... --video 5   # record 5 episodes, exit

See `walls.play` for the full option set (--device, --delay, --video_path).
"""

import argparse
import dataclasses
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import torch

from momentum.env import MomentumWallsVecEnv
from momentum.env_cfg import MomentumVecEnvCfg
from walls.play import WallsPanel, build_runner, make_writer, run_rollout


def load_cfgs(checkpoint: str):
    """Return (agent_dict, env_dict) from <run_dir>/cfg.json, asserting task."""
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint)))
    cfg_path = os.path.join(run_dir, "cfg.json")
    if not os.path.isfile(cfg_path):
        raise SystemExit(f"No cfg.json found at {cfg_path}")
    with open(cfg_path) as f:
        data = json.load(f)
    if "agent" not in data or "env" not in data:
        raise SystemExit(f"{cfg_path} lacks top-level 'agent'/'env' keys.")
    task = data.get("task")
    if task is not None and task != "momentum":
        raise SystemExit(
            f"{cfg_path} is for task {task!r}, not 'momentum'. "
            "Use the matching play script (e.g. python -m walls.play)."
        )
    return data["agent"], data["env"]


# How often (in env steps) the live visualiser windows refresh during play. The
# policy/value maps are static under the frozen checkpoint, so this only paces
# redraws and the intrinsic visitation-count accumulation; a forced first update
# populates them immediately so nothing sits blank.
VIS_UPDATE_EVERY = 25


def attach_visualisers(agent_d: dict, runner, env):
    """Build the momentum spatial visualisers and hook them onto the env.

    Mirrors `train.build_visualisers` for the momentum task — policy + value, plus
    the intrinsic-reward map when RND is on. The replay-buffer visualiser is
    skipped: `build_runner` shrinks the buffer to size 1 for play, so it would be
    empty. Appending to `env._visualisers` lets `env.step` drive `maybe_update`,
    exactly as during training.
    """
    from momentum.visualise import (
        MomentumPolicyVisualiser, MomentumSACValueVisualiser,
        MomentumPPOValueVisualiser, MomentumIntrinsicRewardVisualiser,
        MomentumPPOIntrinsicValueVisualiser,
    )
    algo = str(agent_d.get("algo_name", "")).lower()
    has_intrinsic = getattr(runner.runner, "intrinsic", None) is not None

    visualisers = [MomentumPolicyVisualiser(runner, env, update_every=VIS_UPDATE_EVERY)]
    if algo == "sac":
        visualisers.append(MomentumSACValueVisualiser(runner, env, update_every=VIS_UPDATE_EVERY))
        if has_intrinsic:
            visualisers.append(MomentumIntrinsicRewardVisualiser(runner, env, update_every=VIS_UPDATE_EVERY))
    else:
        visualisers.append(MomentumPPOValueVisualiser(runner, env, update_every=VIS_UPDATE_EVERY))
        if has_intrinsic:
            visualisers.append(MomentumIntrinsicRewardVisualiser(runner, env, update_every=VIS_UPDATE_EVERY))
            visualisers.append(MomentumPPOIntrinsicValueVisualiser(runner, env, update_every=VIS_UPDATE_EVERY))

    env._visualisers.extend(visualisers)
    for v in visualisers:  # populate the maps immediately instead of waiting a cycle
        v.update()
    print(f"[INFO] Attached {len(visualisers)} visualiser(s): "
          f"{', '.join(type(v).__name__ for v in visualisers)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a model_X.pth checkpoint.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--delay", type=float, default=0.03,
                        help="Seconds to pause between animation frames.")
    parser.add_argument("--video", type=int, default=None, metavar="N",
                        help="Record until N episodes have ended, save, then exit.")
    parser.add_argument("--video_path", type=str, default=None,
                        help="Output path for --video (default: <run_dir>/rollout.mp4).")
    parser.add_argument("--visualise", action="store_true",
                        help="Also open the live policy/value visualiser windows.")
    args = parser.parse_args()

    if args.video is not None and args.video < 1:
        raise SystemExit("--video must be a positive integer.")
    if args.visualise and args.video is not None:
        raise SystemExit("--visualise needs a display; it can't be used with headless --video.")

    agent_d, env_d = load_cfgs(args.checkpoint)
    env_cfg = dataclasses.replace(MomentumVecEnvCfg(**env_d), num_envs=1)

    if args.video is not None:
        matplotlib.use("Agg", force=True)  # headless: no display needed to record

    env = MomentumWallsVecEnv(env_cfg, device=args.device)
    runner = build_runner(agent_d, env)
    runner.load_checkpoint(args.checkpoint)
    print(f"[INFO] Loaded {args.checkpoint} (n_walls={env_cfg.n_walls}, "
          f"seed={env_cfg.seed}, momentum={env_cfg.momentum}, "
          f"algo={agent_d.get('algo_name')})")

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
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        hold = max(1, round(fps * 0.5))  # ~0.5s linger on each reached frame
        with writer.saving(fig, out_path, dpi=100):
            completed = run_rollout(runner, env, panel, fig, delay=args.delay,
                                    max_episodes=args.video,
                                    on_frame=writer.grab_frame, reset_hold_frames=hold)
        print(f"[INFO] Saved {completed} episode(s) to {out_path}")
    else:
        if args.visualise:
            attach_visualisers(agent_d, runner, env)
        fig.show()
        run_rollout(runner, env, panel, fig, delay=args.delay)


if __name__ == "__main__":
    main()
