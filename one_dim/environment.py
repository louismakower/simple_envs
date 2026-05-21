from abc import ABC, abstractmethod

import numpy as np
import torch

import one_dim.constants as constants
import one_dim.config as config
from louis_rl.vec_env import SpaceInfo
from .graphics import Graphics


# The environment class defines where the robot starts, where the goal is, and where the obstacle is.
class Environment:

    # Initialisation of a new environment
    def __init__(self):
        if config.ENVIRONMENT_TYPE == "random":
            # Initial state
            self.init_state = np.random.random((1,))
            # Goal state
            self.goal_state = np.random.random((1,))
        elif config.ENVIRONMENT_TYPE == "fixed":
            # Initial state
            self.init_state = np.array([0.9])
            # Goal state
            self.goal_state = np.array([0.2])
        # Set the current state to the initial state
        self.state = self.init_state

    # Reset the environment, i.e. set the state to the initial state
    def reset(self):
        if config.ENVIRONMENT_TYPE == "random":
            self.init_state = np.random.random((1,))
            # Goal state
            self.goal_state = np.random.random((1,))
        self.state = self.init_state

    # Step the environment by executing one action
    def step(self, action):
        next_state = self.dynamics(self.state, action)
        self.state = next_state
        return next_state

    # The environment dynamics, i.e. the transition function
    def dynamics(self, state, action):
        action = action * constants.MAX_ACTION_MAGNITUDE
        # First, clip the action in each dimension
        action = np.clip(action, -constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE)
        # The dynamics is a simple sum
        next_state = state + action
        # Check if the robot is past the boundary
        intersection = self.intersects_perimeter(state, next_state)
        if intersection is not None:
            # If there is an intersection, set the state to just before the point of intersection
            direction = (state - next_state) / np.linalg.norm(state - next_state)
            next_state = intersection + 0.0001 * direction
            return next_state
        # Otherwise, if the robot is inside the environment and not within the obstacle, return the next state as calculated before
        return next_state

    # Function to check if a line intersects the environment perimeter
    # Returns None if the segment doesn’t hit the square’s perimeter
	# Otherwise returns the intersection point
    def intersects_perimeter(self, x1, x2):
        if x2 < constants.ENV_BOUNDS[0]:
            return constants.ENV_BOUNDS[0]
        elif x2 > constants.ENV_BOUNDS[1]:
            return constants.ENV_BOUNDS[1]


class OneDimVecEnv:
    def __init__(self, num_envs: int, max_episode_length: int, device: str = "cpu", headless=False):
        self._num_envs = num_envs
        self._max_episode_length = max_episode_length
        self._device = torch.device(device)
        self._envs = [Environment() for _ in range(num_envs)]
        self._step_counts = torch.zeros(num_envs, dtype=torch.long, device=self._device)
        self.headless = headless
        self._graphics = Graphics() if not headless else None
        # Optional live visualisers (policy, value, ...), attached by the
        # training script. Each must expose a `maybe_update()` method.
        self._visualisers = []

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def max_episode_length(self) -> int:
        return self._max_episode_length

    @property
    def action_space(self) -> SpaceInfo:
        return SpaceInfo(shape=(1,))

    @property
    def observation_space(self) -> dict:
        return {
            "policy": SpaceInfo(shape=(1,)),
            "goal": {"desired_goal": SpaceInfo(shape=(1,))},
        }

    def compute_reward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return (torch.abs(state - goal).squeeze(-1) < constants.GOAL_RADIUS).float()

    def is_terminal(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return torch.abs(state - goal).squeeze(-1) < constants.GOAL_RADIUS
    
    def is_timeout(self):
        return self._step_counts >= self._max_episode_length

    
    @property
    def _goal(self) -> torch.Tensor:
        goals = np.stack([e.goal_state for e in self._envs])  # (N, 1)
        return torch.tensor(goals, dtype=torch.float32, device=self._device)
    
    @property
    def _state_t(self) -> torch.Tensor:
        states = np.stack([e.state for e in self._envs])  # (N, 1)
        return torch.tensor(states, dtype=torch.float32, device=self._device)

    def _get_obs(self) -> dict:
        return {
            "policy": self._state_t,
            "goal": {"desired_goal": self._goal},
            "her": {"position": self._state_t},
        }

    def reset(self) -> tuple[dict, dict]:
        for e in self._envs:
            e.reset()
        self._step_counts.zero_()
        return self._get_obs(), {}

    def step(self, action: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        actions = action.cpu().numpy()  # (N, 1)
        for env, a in zip(self._envs, actions):
            env.step(a)
        self._step_counts += 1

        obs = self._get_obs()
        state = obs["policy"]
        goal = obs["goal"]["desired_goal"]

        rew = self.compute_reward(state, goal)
        term = self.is_terminal(state, goal)
        timeout = self.is_timeout()

        nan = float("nan")
        terminal_obs = {
            "policy": torch.full((self._num_envs, 1), nan, device=self._device),
            "goal": {"desired_goal": torch.full((self._num_envs, 1), nan, device=self._device)},
            "her": {"position": torch.full((self._num_envs, 1), nan, device=self._device)},
        }

        resetted = term | timeout
        if resetted.any():
            terminal_obs["policy"][resetted] = state[resetted].clone()
            terminal_obs["goal"]["desired_goal"][resetted] = goal[resetted].clone()
            terminal_obs["her"]["position"][resetted] = state[resetted].clone()
            for i in range(self._num_envs):
                if resetted[i]:
                    self._envs[i].reset()
                    self._step_counts[i] = 0
        
        if not self.headless:
            self._graphics.draw(self._envs[0], visualisations=[])
        for visualiser in self._visualisers:
            visualiser.maybe_update()
        return self._get_obs(), rew, term, timeout, {"terminal_obs": terminal_obs}
