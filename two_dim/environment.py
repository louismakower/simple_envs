from abc import ABC, abstractmethod

import numpy as np
import pygame
import torch

import two_dim.constants as constants
import two_dim.config as config
from louis_rl.vec_env import SpaceInfo
from .graphics import Graphics


# The environment class defines where the robot starts, where the goal is, and where the obstacle is.
class Environment:

    # Initialisation of a new environment
    def __init__(self):
        # The obstacle is fixed for every environment.
        self.obstacle_pos = np.array([0.55, 0.45])
        self.obstacle_radius = 0.25

        if config.ENVIRONMENT_TYPE == 'fixed':
            # Fixed initial state and goal state
            self.init_state = np.array([0.2, 0.3])
            self.goal_state = np.array([0.9, 0.9])
        # Different initial state and goal state for each episode
        elif config.ENVIRONMENT_TYPE == 'random':
            self._init_random()
        # Set the current state to the initial state
        self.state = self.init_state

    def _sample_free_point(self):
        # Sample a point in the unit square that lies outside the obstacle.
        radius_sum = self.obstacle_radius + constants.ROBOT_RADIUS
        while True:
            point = np.random.uniform(constants.ENV_BOUNDS[0], constants.ENV_BOUNDS[1], size=(2,))
            if np.linalg.norm(point - self.obstacle_pos) > radius_sum:
                return point

    def _init_random(self):
        # Random initial state and goal state, both clear of the obstacle.
        self.init_state = self._sample_free_point()
        self.goal_state = self._sample_free_point()

    # Reset the environment, i.e. set the state to the initial state
    def reset(self):
        if config.ENVIRONMENT_TYPE == "random":
            self._init_random()
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
        intersection = self.intersects_circle(state, next_state, self.obstacle_pos, self.obstacle_radius)
        if intersection:
            # If there is an intersection, set the state to just before the point of intersection
            direction = (state - next_state) / np.linalg.norm(state - next_state)
            next_state = intersection + 0.0001 * direction
            return next_state
        # Check if the robot is past the boundary
        intersection = self.intersects_perimeter(state, next_state)
        if intersection:
            # If there is an intersection, set the state to just before the point of intersection
            direction = (state - next_state) / np.linalg.norm(state - next_state)
            next_state = intersection + 0.0001 * direction
            return next_state
        # Otherwise, if the robot is inside the environment and not within the obstacle, return the next state as calculated before
        return next_state

    # Function to check if a line intersects the circle obstacle
    # Returns None if the line doesn't intersect with the circle
    # Otherwise returns the intersection point
    def intersects_circle(self, p1, p2, center, r):
        (x1, y1), (x2, y2) = p1, p2
        cx, cy = center
        dx, dy = x2 - x1, y2 - y1
        # shift to circle coordinates
        fx, fy = x1 - cx, y1 - cy
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - r * r
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        hits = [(t, (x1 + t * dx, y1 + t * dy)) for t in (t1, t2) if 0 <= t <= 1]
        return min(hits)[1] if hits else None

    # Function to check if a line intersects the environment perimeter
    # Returns None if the segment doesn’t hit the square’s perimeter
	# Otherwise returns the intersection point
    def intersects_perimeter(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        hits = []
        if dx:
            for x in (0.0, 1.0):
                t = (x - x1) / dx
                if 0 <= t <= 1:
                    y = y1 + t * dy
                    if 0 <= y <= 1:
                        hits.append((t, (x, y)))
        if dy:
            for y in (0.0, 1.0):
                t = (y - y1) / dy
                if 0 <= t <= 1:
                    x = x1 + t * dx
                    if 0 <= x <= 1:
                        hits.append((t, (x, y)))
        return min(hits)[1] if hits else None


class TwoDimVecEnv:
    def __init__(self, num_envs: int, max_episode_length: int, device: str = "cpu", headless=False):
        self._num_envs = num_envs
        self._max_episode_length = max_episode_length
        self._device = torch.device(device)
        self._envs = [Environment() for _ in range(num_envs)]
        self._step_counts = torch.zeros(num_envs, dtype=torch.long, device=self._device)
        self.headless = headless
        self._graphics = Graphics() if not headless else None

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
        return SpaceInfo(shape=(2,))

    @property
    def observation_space(self) -> dict:
        return {
            "policy": SpaceInfo(shape=(2,)),
            "goal": {"desired_goal": SpaceInfo(shape=(2,))},
        }

    def compute_reward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return (torch.norm(state - goal, dim=-1) < constants.GOAL_RADIUS).float()

    def is_terminal(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return torch.norm(state - goal, dim=-1) < constants.GOAL_RADIUS
    
    def is_timeout(self):
        return self._step_counts >= self._max_episode_length

    
    @property
    def _goal(self) -> torch.Tensor:
        goals = np.stack([e.goal_state for e in self._envs])  # (N, 2)
        return torch.tensor(goals, dtype=torch.float32, device=self._device)
    
    @property
    def _state_t(self) -> torch.Tensor:
        states = np.stack([e.state for e in self._envs])  # (N, 2)
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
        actions = action.cpu().numpy()  # (N, 2)
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
            "policy": torch.full((self._num_envs, 2), nan, device=self._device),
            "goal": {"desired_goal": torch.full((self._num_envs, 2), nan, device=self._device)},
            "her": {"position": torch.full((self._num_envs, 2), nan, device=self._device)},
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
            # Service the OS event queue so the pygame window actually renders
            # (on macOS the window stays blank/behind until events are pumped).
            pygame.event.pump()
            self._graphics.draw(self._envs[0], visualisations=[])
        return self._get_obs(), rew, term, timeout, {"terminal_obs": terminal_obs}
