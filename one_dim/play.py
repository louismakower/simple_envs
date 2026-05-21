import argparse
import json
from pathlib import Path

import pygame
import torch

from louis_rl.algorithm import RLRunner
from louis_rl.ppo import PPORunnerCfg
from louis_rl.sac import SACRunnerCfg
from one_dim.agents.her_cfg import OneDimHERCfg
from one_dim.environment import OneDimVecEnv
from one_dim.graphics import Graphics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file.")
    parser.add_argument("--max-episode-length", type=int, default=200)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    cfg_path = checkpoint_path.parent.parent / "cfg.json"
    with open(cfg_path) as f:
        cfg_dict = json.load(f)

    algo_name = cfg_dict.get("algo_name", "ppo").lower()
    if algo_name == "ppo":
        cfg = PPORunnerCfg(**cfg_dict)
    elif algo_name == "sac":
        # asdict() flattened the nested OneDimHERCfg into a plain dict; rehydrate it.
        her_cfg = cfg_dict.get("her_cfg")
        if isinstance(her_cfg, dict):
            cfg_dict["her_cfg"] = OneDimHERCfg(**her_cfg)
        cfg = SACRunnerCfg(**cfg_dict)
    else:
        raise ValueError(f"Unknown algo_name in cfg: {algo_name}")

    env = OneDimVecEnv(num_envs=1, max_episode_length=args.max_episode_length)
    runner = RLRunner(env=env, cfg=cfg, log_dir="/tmp/play", writer=None)
    runner.load_checkpoint(args.checkpoint)

    pygame.init()
    graphics = Graphics()

    single_env = env._envs[0]
    obs, _ = env.reset()

    running = True
    with torch.inference_mode():
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action = runner.get_deterministic_action(obs)
            obs, _, _, _, _ = env.step(action)

            graphics.draw(single_env, [])

    pygame.quit()


if __name__ == "__main__":
    main()
