from __future__ import annotations

from dataclasses import dataclass
import torch

from n_dim.env import reward_fn

from louis_rl.her import HERCfg, build_hindsight_goals


def replace_goal_obs(all_obs, goal_obs, policy_obs_dim):
    """Swap the goal slice of a concatenated [policy_obs | goal] observation."""
    without_goal = all_obs[:, :, :policy_obs_dim]
    return torch.cat([without_goal, goal_obs], dim=-1)


def get_her_goals(trajectories, extras):
    obs = trajectories["obs"]  # (T, N, obs_dim)
    next_obs = trajectories["next_obs"]
    action = trajectories["action"]
    her_obs = trajectories["her_obs"]
    her_next_obs = trajectories["her_next_obs"]
    lengths = trajectories["lengths"]
    valid = trajectories["valid"]

    first_pos = her_obs["position"]  # (T, N, n)
    next_pos = her_next_obs["position"]

    policy_obs_dim = extras["policy_obs_dim"]
    goal_radius = extras["goal_radius"]
    n = extras["n"]

    new_goals = build_hindsight_goals(next_pos, lengths, mode=extras["mode"])  # (T, N, n)

    obs = replace_goal_obs(obs, new_goals, policy_obs_dim)
    next_obs = replace_goal_obs(next_obs, new_goals, policy_obs_dim)

    flat_first_pos = first_pos.view(-1, n)
    flat_next_pos = next_pos.view(-1, n)
    flat_goal = new_goals.view(-1, n)

    impossible_first_state = torch.norm(flat_first_pos - flat_goal, dim=-1) < goal_radius
    valid = valid & ~impossible_first_state
    distances = torch.norm(flat_next_pos - flat_goal, dim=-1)
    within = distances < goal_radius
    reward = reward_fn(flat_next_pos, flat_goal, goal_radius, n)

    return {
        "obs": obs.view(-1, obs.shape[2])[valid],
        "action": action.view(-1, action.shape[2])[valid],
        "reward": reward.unsqueeze(-1)[valid],
        "next_obs": next_obs.view(-1, next_obs.shape[2])[valid],
        "done": within.unsqueeze(-1)[valid],
        "distances": distances[valid],
    }


@dataclass
class NDimHERCfg(HERCfg):
    mode: str = "future"
    n: int = 2
    goal_radius: float = 0.1
    k: int = 4

    def __post_init__(self):
        self.policy_obs_dim = None

    def get_hindsight_transitions(self, trajectories, extras={}):
        extras.update({
            "mode": self.mode,
            "goal_radius": self.goal_radius,
            "n": self.n,
            "policy_obs_dim": self.policy_obs_dim,
        })
        return get_her_goals(trajectories, extras)
