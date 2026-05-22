"""Hindsight Experience Replay config for the 2D environment.

The observation handed to the policy is the concatenation [policy_obs | desired_goal]
(see base_runner.add_goal_obs), so relabeling a transition just means swapping the
goal slice. The reward is sparse: 1 when the achieved goal lands within GOAL_RADIUS
of the desired goal, matching TwoDimVecEnv.compute_reward / is_terminal.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from louis_rl.her import HERCfg, build_hindsight_goals
import two_dim.constants as constants


def compute_reward(position, desired_goal, goal_radius):
    """Sparse reward: 1.0 when achieved is within goal_radius of desired, else 0.0."""
    return (torch.norm(position - desired_goal, dim=-1) < goal_radius).float()


def replace_goal_obs(all_obs, goal_obs, policy_obs_dim):
    """Swap the goal slice of a concatenated [policy_obs | goal] observation."""
    without_goal = all_obs[:, :, :policy_obs_dim]
    return torch.cat([without_goal, goal_obs], dim=-1)


def get_her_goals(trajectories, extras):
    obs = trajectories["obs"]  # (T, N, obs_dim)
    next_obs = trajectories["next_obs"]  # (T, N, obs_dim)
    action = trajectories["action"]  # (T, N, action_dim)
    her_obs = trajectories["her_obs"]  # dict[str, Tensor(T, N, ...)]
    her_next_obs = trajectories["her_next_obs"]  # dict[str, Tensor(T, N, ...)]
    lengths = trajectories["lengths"]  # (N,) — valid steps per env
    valid = trajectories["valid"]  # (T * N,) bool

    first_pos = her_obs["position"]  # (T, N, 2)
    next_pos = her_next_obs["position"]  # (T, N, 2)

    policy_obs_dim = extras["policy_obs_dim"]
    goal_radius = extras["goal_radius"]

    # Hindsight goals are drawn from the next-achieved-goal trajectory.
    new_goals = build_hindsight_goals(next_pos, lengths, mode=extras["mode"])  # (T, N, 2)

    obs = replace_goal_obs(obs, new_goals, policy_obs_dim)
    next_obs = replace_goal_obs(next_obs, new_goals, policy_obs_dim)

    flat_first_pos = first_pos.view(-1, 2)
    flat_next_pos = next_pos.view(-1, 2)
    flat_goal = new_goals.view(-1, 2)

    impossible_first_state = torch.norm(flat_first_pos - flat_goal, dim=-1) < goal_radius  # these states would have already terminated
    valid = valid & ~impossible_first_state
    distances = torch.norm(flat_next_pos - flat_goal, dim=-1)  # (T * N,)
    # Recompute reward and termination against the hindsight goal — the original
    # term flag refers to the real goal and is meaningless after relabelling.
    within = distances < goal_radius
    reward = within.float()

    return {
        "obs": obs.view(-1, obs.shape[2])[valid],
        "action": action.view(-1, action.shape[2])[valid],
        "reward": reward.unsqueeze(-1)[valid],
        "next_obs": next_obs.view(-1, next_obs.shape[2])[valid],
        "done": within.unsqueeze(-1)[valid],
        "distances": distances[valid],
    }


@dataclass
class TwoDimHERCfg(HERCfg):
    mode: str = "future"
    goal_radius: float = constants.GOAL_RADIUS
    k: int = 4

    def __post_init__(self):
        # Set by SACRunner after env construction; not a config field.
        self.policy_obs_dim = None

    def get_hindsight_transitions(self, trajectories, extras={}):
        extras.update({"mode": self.mode, "goal_radius": self.goal_radius, "policy_obs_dim": self.policy_obs_dim})
        return get_her_goals(trajectories, extras)
