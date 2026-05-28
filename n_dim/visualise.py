"""Live visualisations for the n-dim env.

`PolicyVisualiser` shows a quiver of the policy's first two action components
over a grid of (s_0, s_1). Other state and goal dims are pinned to 0.5. The
2x2 figure mirrors the two_dim convention: one subplot per fixed first-2-dim
goal corner from `GOALS`. For n=1 the figure collapses to a 1x2 grid and the
second state dimension is padded by tiling s_0.

`SACValueVisualiser` / `PPOValueVisualiser` show value vs. distance-to-goal as
a scatter over randomly sampled (state, goal) pairs in [0,1]^n.
"""

import math

import numpy as np
import torch
import matplotlib.pyplot as plt


# First-2-dim goal corners (permutations of (0.1, 0.9)). Other dims = 0.5.
GOALS = [(0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)]
# Goals when n == 1: just two positions along the single axis.
GOALS_1D = [0.1, 0.9]
# Pinned value for non-plotted state/goal dimensions (n >= 3).
FILL_VALUE = 0.5


class PolicyVisualiser:
    QUIVER_RES = 21
    MAGNIFY = 1.0

    def __init__(self, runner, env, update_every: int = 1):
        self._runner = runner
        self._env = env
        self._n = env.n
        self._max_step_size = env.max_step_size
        self._update_every = update_every
        self._step = 0
        self._device = env.device
        self._setup_figure()

    def _setup_figure(self):
        plt.ion()
        if self._n == 1:
            self._fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            self._goals = GOALS_1D
        else:
            self._fig, axes = plt.subplots(2, 2, figsize=(10, 9))
            self._goals = GOALS
        self._axes = list(np.atleast_1d(axes).ravel())
        try:
            self._fig.canvas.manager.set_window_title("Policy (first 2 dims)")
        except Exception:
            pass

        res = self.QUIVER_RES
        axis = np.linspace(0.0, 1.0, res)
        self._X, self._Y = np.meshgrid(axis, axis)
        zero = np.zeros((res, res))
        self._quivers = []
        for ax, goal in zip(self._axes, self._goals):
            q = ax.quiver(
                self._X, self._Y, zero, zero,
                angles="xy", scale_units="xy", scale=1.0, width=0.004,
            )
            self._quivers.append(q)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_aspect("equal")
            ax.set_title(f"goal = {goal}")
            ax.set_xlabel("state dim 0")
            ax.set_ylabel("state dim 1" if self._n >= 2 else "(padded)")

        self._fig.suptitle("Policy: deterministic action (first 2 dims)")
        self._fig.tight_layout()
        self._fig.show()

    @torch.no_grad()
    def _action(self, states, goals):
        obs = {"policy": states, "goal": {"desired_goal": goals}}
        return self._runner.get_deterministic_action(obs)

    def maybe_update(self):
        self._step += 1
        if self._step % self._update_every == 0:
            self.update()

    def _build_states(self, x_flat, y_flat):
        # x_flat, y_flat: (res*res,) tensors on device, in [0, 1].
        if self._n == 1:
            return x_flat.unsqueeze(-1)  # (res*res, 1)
        cols = [x_flat, y_flat]
        if self._n > 2:
            fill = torch.full((x_flat.numel(), self._n - 2), FILL_VALUE, device=self._device)
            return torch.cat([torch.stack(cols, dim=1), fill], dim=1)
        return torch.stack(cols, dim=1)

    def _build_goals(self, goal_corner, batch_size):
        if self._n == 1:
            g = torch.tensor([goal_corner], dtype=torch.float32, device=self._device)
            return g.expand(batch_size, 1)
        first_two = torch.tensor(goal_corner, dtype=torch.float32, device=self._device)
        if self._n == 2:
            return first_two.expand(batch_size, 2)
        rest = torch.full((self._n - 2,), FILL_VALUE, device=self._device)
        full = torch.cat([first_two, rest])
        return full.expand(batch_size, self._n)

    def update(self):
        res = self.QUIVER_RES
        axis = torch.linspace(0.0, 1.0, res, device=self._device)
        y_grid, x_grid = torch.meshgrid(axis, axis, indexing="ij")
        x_flat = x_grid.reshape(-1)
        y_flat = y_grid.reshape(-1)
        states = self._build_states(x_flat, y_flat)

        for quiver, goal in zip(self._quivers, self._goals):
            goals = self._build_goals(goal, states.shape[0])
            action = self._action(states, goals) * self._max_step_size
            action = (action * self.MAGNIFY).cpu().numpy()
            if self._n == 1:
                u = action[:, 0].reshape(res, res)
                v = np.zeros_like(u)
            else:
                u = action[:, 0].reshape(res, res)
                v = action[:, 1].reshape(res, res)
            quiver.set_UVC(u, v)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()


class _BaseValueVisualiser:
    """Value vs distance-to-goal scatter. Subclasses implement `_value`."""

    NUM_SAMPLES = 1000

    _title = "Value vs distance to goal"
    _ylabel = "V"

    def __init__(self, env, update_every: int = 20):
        self._env = env
        self._n = env.n
        self._update_every = update_every
        self._step = 0
        self._device = env.device
        self._setup_figure()

    def _setup_figure(self):
        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(6, 5))
        try:
            self._fig.canvas.manager.set_window_title(self._title)
        except Exception:
            pass
        self._scatter = self._ax.scatter([], [], s=4, alpha=0.4)
        self._ax.set_xlim(0.0, math.sqrt(self._n))
        self._ax.set_xlabel("distance to goal")
        self._ax.set_ylabel(self._ylabel)
        self._ax.set_title(self._title)
        self._fig.tight_layout()
        self._fig.show()

    @torch.no_grad()
    def _value(self, states, goals):
        raise NotImplementedError

    def maybe_update(self):
        self._step += 1
        if self._step % self._update_every == 0:
            self.update()

    def update(self):
        states = torch.rand(self.NUM_SAMPLES, self._n, device=self._device)
        goals = torch.rand(self.NUM_SAMPLES, self._n, device=self._device)
        dists = torch.norm(states - goals, dim=-1).cpu().numpy()
        values = self._value(states, goals).cpu().numpy()

        self._scatter.set_offsets(np.column_stack([dists, values]))
        self._ax.set_ylim(float(values.min()), float(values.max()) + 1e-6)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()


class SACValueVisualiser(_BaseValueVisualiser):
    NUM_ACTION_SAMPLES = 16
    _title = "SAC value vs distance to goal"
    _ylabel = "mean min(q1, q2) over sampled actions"

    def __init__(self, runner, env, update_every: int = 20):
        self._sac = runner.runner
        super().__init__(env, update_every)

    @torch.no_grad()
    def _value(self, states, goals):
        sac = self._sac
        sac._set_eval()
        obs = {"policy": states, "goal": {"desired_goal": goals}}
        obs_n = sac.obs_norm(sac.add_goal_obs(obs))
        dist = sac.policy.dist(obs_n)
        v = torch.zeros(obs_n.shape[0], 1, device=self._device)
        for _ in range(self.NUM_ACTION_SAMPLES):
            act = dist.sample()
            q_input = sac._q_input(obs_n, act)
            q = torch.minimum(sac.q1.network(q_input), sac.q2.network(q_input))
            v += q
        return (v / self.NUM_ACTION_SAMPLES).squeeze(-1)


class PPOValueVisualiser(_BaseValueVisualiser):
    _title = "PPO value vs distance to goal"
    _ylabel = "V"

    def __init__(self, runner, env, update_every: int = 20):
        self._ppo = runner.runner
        super().__init__(env, update_every)

    @torch.no_grad()
    def _value(self, states, goals):
        ppo = self._ppo
        ppo.v.eval()
        obs = {"policy": states, "goal": {"desired_goal": goals}}
        return ppo.v(ppo.add_goal_obs(obs)).squeeze(-1)
