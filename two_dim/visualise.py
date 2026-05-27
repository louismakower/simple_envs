"""Live visualisations over the 2D state space for fixed goals.

The 2D environment has a 2D state *and* a 2D goal, so a function of (state,
goal) lives in 4D and cannot be shown on a single plot. Instead each visualiser
fixes the goal at one of four corner positions -- the permutations of
(0.1, 0.9) -- and shows the function over the 2D state unit square in its own
subplot. Everything is computed from the live networks held in memory --
nothing is loaded from a checkpoint.

`Visualiser` shows the deterministic policy action as a field of little arrows
(a quiver plot): the arrow at a state points the way the policy moves. It works
for both SAC and PPO -- the action comes from the runner's
`get_deterministic_action`.

`ValueVisualiser` shows the SAC state-value function V(state) as a heatmap on
the same axes, as in the 1D version. It is SAC-only -- it queries the twin
critics directly (see that class for detail).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

import two_dim.constants as constants


# The four fixed goal positions: permutations of (0.1, 0.9).
GOALS = [(0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)]


def _draw_obstacle(ax, env):
    """Outline the (fixed) circular obstacle on a subplot."""
    base = env._envs[0]
    ax.add_patch(plt.Circle(
        tuple(base.obstacle_pos), base.obstacle_radius,
        fill=False, edgecolor="black", linestyle="--", linewidth=1.0, zorder=3,
    ))


class Visualiser:
    """Live visualisation of the deterministic policy over the 2D state space.

    One 2x2 figure: each subplot fixes the goal at a corner from `GOALS` and
    draws the policy action at a grid of states as a quiver field. Works for
    both SAC and PPO.
    """

    # Resolution of the (square) state grid sampled for the quiver field.
    QUIVER_RES = 21
    # Arrows are drawn as the true per-step displacement (action *
    # MAX_ACTION_MAGNITUDE) magnified by this factor, so the field is readable
    # while still showing relative magnitude differences.
    MAGNIFY = 4.0

    def __init__(self, runner, env, update_every: int = 1):
        # `runner` is the RLRunner; `env` is the TwoDimVecEnv.
        self._runner = runner
        self._env = env
        self._update_every = update_every
        self._step = 0
        self._device = env.device
        self._setup_figure()

    def _setup_figure(self):
        plt.ion()
        self._fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        self._axes = list(axes.ravel())
        try:
            self._fig.canvas.manager.set_window_title("Policy (2D state space)")
        except Exception:
            pass

        res = self.QUIVER_RES
        axis = np.linspace(0.0, 1.0, res)
        # X[i, j] = axis[j] (state x), Y[i, j] = axis[i] (state y).
        self._X, self._Y = np.meshgrid(axis, axis)
        zero = np.zeros((res, res))
        self._quivers = []
        for ax, goal in zip(self._axes, GOALS):
            q = ax.quiver(
                self._X, self._Y, zero, zero,
                angles="xy", scale_units="xy", scale=1.0, width=0.004,
            )
            self._quivers.append(q)
            _draw_obstacle(ax, self._env)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_aspect("equal")
            ax.set_title(f"goal = {goal}")
            ax.set_xlabel("state x")
            ax.set_ylabel("state y")

        self._fig.suptitle("Policy: deterministic action")
        self._fig.tight_layout()
        self._fig.show()

    @torch.no_grad()
    def _action(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        """Deterministic policy action at (state, goal) pairs.

        `states`, `goals` are (N, 2) tensors. Returns a (N, 2) action tensor.
        """
        obs = {"policy": states, "goal": {"desired_goal": goals}}
        return self._runner.get_deterministic_action(obs)

    def maybe_update(self):
        """Refresh the matplotlib figure every `update_every` steps."""
        self._step += 1
        if self._step % self._update_every == 0:
            self.update()

    def update(self):
        res = self.QUIVER_RES
        axis = torch.linspace(0.0, 1.0, res, device=self._device)
        # rows -> state y, cols -> state x (matches the np.meshgrid above).
        y_grid, x_grid = torch.meshgrid(axis, axis, indexing="ij")
        states = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=1)

        for quiver, goal in zip(self._quivers, GOALS):
            goal_t = torch.tensor(goal, dtype=torch.float32, device=self._device)
            goals = goal_t.expand(states.shape[0], 2)
            action = self._action(states, goals) * constants.MAX_ACTION_MAGNITUDE
            action = (action * self.MAGNIFY).cpu().numpy().reshape(res, res, 2)
            quiver.set_UVC(action[..., 0], action[..., 1])

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()


class _BaseValueVisualiser:
    """Shared figure + update loop for value heatmaps over the 2D state.

    One 2x2 figure: each subplot fixes the goal at a corner from `GOALS` and
    draws V(state) as a heatmap. Subclasses implement `_value(states, goals)`
    to plug in the algorithm's value function and set `_title` / `_clabel`.
    All four subplots share one autoscaled colour range each update so values
    are directly comparable across goals.
    """

    # Resolution of the (square) state grid evaluated for the heatmap.
    HEATMAP_RES = 80

    # Subclasses override these.
    _title = "Value"
    _clabel = "V"

    def __init__(self, env, update_every: int = 20):
        self._env = env
        self._update_every = update_every
        self._step = 0
        self._device = env.device
        self._setup_figure()

    def _setup_figure(self):
        plt.ion()
        self._fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        self._axes = list(axes.ravel())
        try:
            self._fig.canvas.manager.set_window_title("Value (2D state space)")
        except Exception:
            pass

        blank = np.zeros((self.HEATMAP_RES, self.HEATMAP_RES))
        self._ims = []
        for ax, goal in zip(self._axes, GOALS):
            # Colour range autoscales (shared) each update -- V has no fixed bounds.
            im = ax.imshow(
                blank, origin="lower", extent=(0.0, 1.0, 0.0, 1.0),
                aspect="equal", cmap="viridis",
            )
            self._ims.append(im)
            _draw_obstacle(ax, self._env)
            ax.set_title(f"goal = {goal}")
            ax.set_xlabel("state x")
            ax.set_ylabel("state y")

        self._fig.suptitle(self._title)
        # One shared colorbar across all four subplots.
        self._fig.colorbar(
            self._ims[0], ax=self._axes, label=self._clabel, fraction=0.046,
        )
        self._fig.show()

    @torch.no_grad()
    def _value(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def maybe_update(self):
        """Refresh the matplotlib figure every `update_every` steps."""
        self._step += 1
        if self._step % self._update_every == 0:
            self.update()

    def update(self):
        res = self.HEATMAP_RES
        axis = torch.linspace(0.0, 1.0, res, device=self._device)
        # rows -> state y, cols -> state x (matches imshow origin="lower").
        y_grid, x_grid = torch.meshgrid(axis, axis, indexing="ij")
        states = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=1)

        grids = []
        for goal in GOALS:
            goal_t = torch.tensor(goal, dtype=torch.float32, device=self._device)
            goals = goal_t.expand(states.shape[0], 2)
            value = self._value(states, goals)
            grids.append(value.reshape(res, res).cpu().numpy())

        # Shared colour range so values are comparable across goals.
        vmin = min(float(g.min()) for g in grids)
        vmax = max(float(g.max()) for g in grids)
        for im, grid in zip(self._ims, grids):
            im.set_data(grid)
            im.set_clim(vmin, vmax)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()


class SACValueVisualiser(_BaseValueVisualiser):
    """SAC value heatmap.

    SAC's critic Q(state, goal, action) cannot be shown directly on a 2D
    heatmap, so it is collapsed to V(state, goal) = mean over sampled policy
    actions of min(q1, q2). Reaches into the SACRunner's twin critics,
    observation normaliser and stochastic policy.
    """

    # Policy action samples averaged per grid cell to estimate V.
    NUM_SAMPLES = 16

    _title = "Value: mean min(q1, q2) over sampled actions"
    _clabel = "Q (scaled-reward units)"

    def __init__(self, runner, env, update_every: int = 20):
        # `runner.runner` is the SACRunner whose critics/policy/normaliser
        # we query directly.
        self._sac = runner.runner
        super().__init__(env, update_every)

    @torch.no_grad()
    def _value(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        sac = self._sac
        sac._set_eval()
        obs = {"policy": states, "goal": {"desired_goal": goals}}
        obs_n = sac.obs_norm(sac.add_goal_obs(obs))
        dist = sac.policy.dist(obs_n)
        v = torch.zeros(obs_n.shape[0], 1, device=self._device)
        for _ in range(self.NUM_SAMPLES):
            act = dist.sample()
            q_input = sac._q_input(obs_n, act)
            q = torch.minimum(sac.q1.network(q_input), sac.q2.network(q_input))
            v += q
        return (v / self.NUM_SAMPLES).squeeze(-1)


class PPOValueVisualiser(_BaseValueVisualiser):
    """PPO value heatmap.

    PPO trains a scalar state-value network V(state, goal) directly, so the
    heatmap is just that network evaluated on the (state, goal) grid -- no
    action sampling, no observation normalisation.
    """

    _title = "Value: V(state, goal)"
    _clabel = "V"

    def __init__(self, runner, env, update_every: int = 20):
        # `runner.runner` is the PPORunner whose value network we query directly.
        self._ppo = runner.runner
        super().__init__(env, update_every)

    @torch.no_grad()
    def _value(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        ppo = self._ppo
        ppo.v.eval()
        obs = {"policy": states, "goal": {"desired_goal": goals}}
        return ppo.v(ppo.add_goal_obs(obs)).squeeze(-1)
