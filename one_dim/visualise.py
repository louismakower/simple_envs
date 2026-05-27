"""Live visualisations over the (state, goal) unit square.

The 1D environment has a 1D state and a 1D goal, so functions of (state, goal)
are just surfaces over the 2D unit square, shown as live matplotlib heatmaps --
x-axis is the state, y-axis is the goal. Everything is computed from the live
networks held in memory -- nothing is loaded from a checkpoint.

`Visualiser` shows the deterministic policy action. It works for both SAC and
PPO: the action comes from the runner's `get_deterministic_action`, so neither
the network internals nor the algorithm need to be known here.

`ValueVisualiser` shows the SAC state-value function V(state, goal). It is
SAC-only -- it queries the twin critics directly (see that class for detail).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt


class Visualiser:
    # Resolution of the (state, goal) grid evaluated for the heatmap.
    HEATMAP_RES = 80

    def __init__(self, runner, env, update_every: int = 1):
        # `runner` is the RLRunner; `env` is the OneDimVecEnv.
        self._runner = runner
        self._env = env
        self._update_every = update_every
        self._step = 0
        self._device = env.device
        self._setup_figure()

    def _setup_figure(self):
        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(6, 5))
        try:
            self._fig.canvas.manager.set_window_title("Policy (state-goal space)")
        except Exception:
            pass

        blank = np.zeros((self.HEATMAP_RES, self.HEATMAP_RES))
        # Colour scale fixed to [-1, 1]; out-of-range actions just saturate.
        self._im = self._ax.imshow(
            blank, origin="lower", extent=(0.0, 1.0, 0.0, 1.0), aspect="auto",
            cmap="coolwarm", vmin=-1.0, vmax=1.0,
        )
        self._ax.set_title("Policy: deterministic action")
        self._ax.set_xlabel("state")
        self._ax.set_ylabel("goal")
        self._fig.colorbar(self._im, ax=self._ax, label="action")


        self._fig.tight_layout()
        self._fig.show()

    @torch.no_grad()
    def _action(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        """Deterministic policy action at (state, goal) pairs.

        `states`, `goals` are (N, 1) tensors. Returns a (N,) action tensor.
        """
        obs = {"policy": states, "goal": {"desired_goal": goals}}
        return self._runner.get_deterministic_action(obs).squeeze(-1)

    def maybe_update(self):
        """Refresh the matplotlib heatmap every `update_every` steps."""
        self._step += 1
        if self._step % self._update_every == 0:
            self.update()

    def update(self):
        res = self.HEATMAP_RES
        axis = torch.linspace(0.0, 1.0, res, device=self._device)
        # rows -> goal, cols -> state (matches imshow origin="lower").
        goal_grid, state_grid = torch.meshgrid(axis, axis, indexing="ij")
        action = self._action(state_grid.reshape(-1, 1), goal_grid.reshape(-1, 1))
        self._im.set_data(action.reshape(res, res).cpu().numpy())

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()


class _BaseValueVisualiser:
    """Shared figure + update loop for value heatmaps over (state, goal).

    Subclasses implement `_value(states, goals)` to plug in the algorithm's
    value function and set `_title` / `_clabel`. The colour range autoscales
    each update -- V has no fixed bounds.
    """

    # Resolution of the (state, goal) grid evaluated for the heatmap.
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
        self._fig, self._ax = plt.subplots(figsize=(6, 5))
        try:
            self._fig.canvas.manager.set_window_title("Value (state-goal space)")
        except Exception:
            pass

        blank = np.zeros((self.HEATMAP_RES, self.HEATMAP_RES))
        self._im = self._ax.imshow(
            blank, origin="lower", extent=(0.0, 1.0, 0.0, 1.0), aspect="auto",
            cmap="viridis",
        )
        self._ax.set_title(self._title)
        self._ax.set_xlabel("state")
        self._ax.set_ylabel("goal")
        self._fig.colorbar(self._im, ax=self._ax, label=self._clabel)

        self._fig.tight_layout()
        self._fig.show()

    @torch.no_grad()
    def _value(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def maybe_update(self):
        """Refresh the matplotlib heatmap every `update_every` steps."""
        self._step += 1
        if self._step % self._update_every == 0:
            self.update()

    def update(self):
        res = self.HEATMAP_RES
        axis = torch.linspace(0.0, 1.0, res, device=self._device)
        # rows -> goal, cols -> state (matches imshow origin="lower").
        goal_grid, state_grid = torch.meshgrid(axis, axis, indexing="ij")
        value = self._value(state_grid.reshape(-1, 1), goal_grid.reshape(-1, 1))
        data = value.reshape(res, res).cpu().numpy()
        self._im.set_data(data)
        self._im.set_clim(float(data.min()), float(data.max()))

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()


class SACValueVisualiser(_BaseValueVisualiser):
    """SAC value heatmap over (state, goal).

    SAC's critic Q(state, goal, action) has a 3D input, so it cannot be shown
    directly. This collapses it to V(state, goal) = mean over sampled policy
    actions of min(q1, q2). Reaches into the SACRunner's twin critics,
    observation normaliser and stochastic policy. The value is the raw critic
    output, i.e. in scaled-reward units (the critic is trained on normalised
    rewards).
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
    """PPO value heatmap over (state, goal).

    PPO trains a scalar state-value network V(state, goal) directly, so the
    heatmap is just that network evaluated on the grid -- no action sampling,
    no observation normalisation.
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
