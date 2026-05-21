"""Live visualisation of the learned policy.

The 1D environment has a 1D state and a 1D goal, so the deterministic policy is
just a function over the 2D (state, goal) unit square. This module queries the
in-memory policy during training and shows it as a live matplotlib heatmap --
x-axis is the state, y-axis is the goal, colour is the action.

It works for both SAC and PPO: the action is obtained via the runner's
`get_deterministic_action`, so neither the network internals nor the algorithm
(squashed/normalised for SAC, raw Gaussian mean for PPO) need to be known here.
Everything is computed from the live policy held in memory -- nothing is loaded
from a checkpoint.
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
