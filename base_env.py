import torch
from louis_rl.vec_env import SpaceInfo, VecEnv
from louis_rl.algos.go_explore import GoExploreVecEnv


class GoalReachVecEnv(VecEnv):
    """Shared skeleton for the simple goal-reaching vec envs.

    A point agent moves in the unit hypercube ``[0, 1] ** dim`` towards a goal,
    terminating inside ``goal_radius`` with a sparse reward. Subclasses set the
    dimensionality and override a small set of hooks:

      - ``_create_goals(num)`` / ``_sample_starts(num)``  (required)
      - ``_starts_valid(starts, goals)``  (default: always valid)
      - ``_transition(state, scaled_action)``  (default: clamp to bounds)
    """

    def __init__(
            self,
            dim: int,
            num_envs: int,
            max_ep_len: int,
            goal_radius: float,
            max_step_size: float,
            goal_dynamics: bool,
            device="cpu",
    ):
        self.dim = dim
        self._num_envs = num_envs
        self.max_ep_len = max_ep_len
        self.goal_radius = goal_radius
        self.max_step_size = max_step_size
        self.goal_dynamics = goal_dynamics
        self._device = device

        self.goal = torch.empty(size=(num_envs, dim), device=device)
        self.state = torch.empty(size=(num_envs, dim), device=device)
        self.ep_counters = torch.zeros(size=(num_envs,), device=device)

        self._visualisers = []

        self.reset()
        self.randomise_ep_counters()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def max_episode_length(self) -> int:
        return self.max_ep_len

    @property
    def action_space(self) -> SpaceInfo:
        return SpaceInfo(shape=(self.dim,))

    @property
    def observation_space(self) -> dict:
        return {
            "policy": SpaceInfo(shape=(self.dim,)),
            "goal": {"desired_goal": SpaceInfo(shape=(self.dim,))},
            "rnd": SpaceInfo(shape=(self.dim,))
        }

    def step(self, action):
        clipped_action = torch.clip(action, -1., 1.)
        scaled_action = clipped_action * self.max_step_size

        next_state = self._transition(self.state, scaled_action)
        self.ep_counters += 1

        rew = self.compute_rew(next_state, self.goal)
        dist_to_goal = _dist(next_state, self.goal).mean()
        term = self.compute_term(next_state)
        timeout = self.compute_timeout()

        to_reset = term | timeout
        terminal_obs = self.get_terminal_obs(
            next_state,
            self.goal,
            env_ids=to_reset,
        )

        self.state[~to_reset] = next_state[~to_reset]

        # reset fn returns all obs
        obs, _ = self.reset(env_ids=to_reset)
        extras = {
            "terminal_obs": terminal_obs,
            "log": {"metrics/dist_to_goal": dist_to_goal},
            }

        for visualiser in self._visualisers:
            visualiser.maybe_update()

        return obs, rew, term, timeout, extras

    def get_obs(self, state, goal):
        return {
            "policy": state.clone(),
            "goal": {"desired_goal": goal.clone()},
            "her": {"position": state.clone()},
            "rnd": state.clone(),
        }

    def get_terminal_obs(self, state, goal, env_ids):
        nan = float("nan")
        terminal_state = torch.where(env_ids.unsqueeze(-1), state, nan)
        terminal_goal = torch.where(env_ids.unsqueeze(-1), goal, nan)
        return self.get_obs(terminal_state, terminal_goal)

    def reset(self, env_ids: torch.Tensor = None):
        # reset all if no env ids are given
        if env_ids is None:
            env_ids = torch.ones(size=(self.num_envs,), dtype=torch.bool, device=self._device)

        num_reset_envs = int(torch.sum(env_ids))
        new_goals = self._create_goals(num_reset_envs)
        new_starts = self._sample_starts(num_reset_envs)

        valid = self._starts_valid(new_starts, new_goals)
        while not valid.all():
            new_starts[~valid] = self._sample_starts(int((~valid).sum()))
            valid = self._starts_valid(new_starts, new_goals)

        self.goal[env_ids] = new_goals
        self.state[env_ids] = new_starts
        self.ep_counters[env_ids] = 0

        return self.get_obs(self.state, self.goal), {}

    def compute_rew(self, state, goal):
        return reward_fn(state, goal, self.goal_radius, goal_dynamics=self.goal_dynamics)

    def compute_term(self, state):
        if self.goal_dynamics:
            return (_dist(state, self.goal) < self.goal_radius)
        else:
            return torch.zeros(state.shape[0], device=self.device, dtype=torch.bool)

    def compute_timeout(self):
        return self.ep_counters >= self.max_ep_len

    def randomise_ep_counters(self):
        self.ep_counters = (torch.rand(self.num_envs, device=self._device) * self.max_ep_len).long()

    # --- hooks ---------------------------------------------------------------

    def _transition(self, state, scaled_action):
        # default dynamics: move and clamp to the unit cube
        return torch.clamp(state + scaled_action, 0., 1.)

    def _starts_valid(self, starts, goals):
        # default: every sampled start is acceptable
        return torch.ones(size=(starts.shape[0],), dtype=torch.bool, device=self._device)

    def _create_goals(self, num_goals=None):
        raise NotImplementedError

    def _sample_starts(self, num):
        raise NotImplementedError

class GoalReachGoExploreVecEnv(GoExploreVecEnv, GoalReachVecEnv):
    def snapshot_state(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        return self.state[env_ids]

    def restore_state(self, env_ids, snapshots):
        if env_ids is None:
            env_ids = slice(None)
        self.state[env_ids] = snapshots
        # full episode exploration on restore
        self.ep_counters[env_ids] = 0

    def terminal_snapshot(self, extras):
        return extras["terminal_obs"]["policy"]


def reward_fn(state, goal, goal_radius, goal_dynamics=True):
    reached = (_dist(state, goal) < goal_radius).float()
    neg_step = torch.full_like(reached, -0.01)
    return reached + neg_step if goal_dynamics else neg_step


def _dist(pos1, pos2):
    return torch.norm(
        pos1 - pos2,
        dim=-1,
    )
