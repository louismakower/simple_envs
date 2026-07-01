from __future__ import annotations

import torch
from louis_rl.vec_env import SpaceInfo

from walls.env import WallsVecEnv
from momentum.env_cfg import MomentumVecEnvCfg

class MomentumWallsVecEnv(WallsVecEnv):
    cfg: MomentumVecEnvCfg

    def __init__(
            self,
            cfg: MomentumVecEnvCfg,
            device="cpu",
    ):
        self.momentum = cfg.momentum
        self.last_action = torch.zeros(size=(cfg.num_envs, 2), device=device)
        super().__init__(cfg, device)

    def _transition(self, state, scaled_action):
        momentum_action = (1 - self.momentum) * scaled_action + self.momentum * self.last_action
        next_state = super()._transition(state, scaled_action=momentum_action)

        # components that didn't move as per the action hit a wall or env end
        # zero the momentum in that direction
        blocked = ((next_state - state) - momentum_action).abs() > 1e-6
        self.last_action = torch.where(blocked, torch.zeros_like(momentum_action), momentum_action)
        return next_state

    def reset(self, env_ids: torch.Tensor = None):
        out = super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self.last_action[env_ids] = 0
        return out

    def restore_state(self, env_ids, snapshots):
        super().restore_state(env_ids, snapshots)
        if env_ids is None:
            env_ids = slice(None)
        self.last_action[env_ids] = 0

    @property
    def observation_space(self) -> dict:
        # avoid it being partially observable
        # obs becomes [position, velocity]
        space = super().observation_space
        space["policy"] = SpaceInfo(shape=(2 * self.dim,))
        # rnd gets velocity too so intrinsic can filter based on it
        space["rnd"] = SpaceInfo(shape=(2 * self.dim,))
        return space

    def get_obs(self, state, goal):
        obs = super().get_obs(state, goal)
        obs["policy"] = torch.cat([obs["policy"], self.last_action], dim=-1)
        obs["rnd"] = torch.cat([obs["rnd"], self.last_action], dim=-1)
        return obs
