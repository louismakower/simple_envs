"""Live spatial visualisations for the 2D walls env.

Everything is plotted over the physical (x, y) maze so collisions and the
learned solution are directly legible. Each spatial axis overlays the frozen
wall/hole geometry (cyan) and, where relevant, the pinned goal (star).

`WallsPolicyVisualiser` — 1x3 quiver of the deterministic action over an
(x, y) grid, one subplot per pinned goal y in `GOAL_YS` (all at x=1).

`WallsSACValueVisualiser` / `WallsPPOValueVisualiser` — 1x3 heatmaps of the
value over an (x, y) grid, one per pinned goal. SAC value is the mean of
min(q1, q2) over sampled actions (same definition as the n_dim visualiser);
PPO value is v(s, g). The colour scale is shared across the row so the three
goals are directly comparable.

`WallsBufferVisualiser` (SAC-only) — 1x2 (x, y) histograms of replay-buffer
transitions binned by state position: mean reward and transition count.
Marginalised over everything in the buffer, including HER-relabelled goals.

`WallsIntrinsicRewardVisualiser` (SAC + RND only) — a single (x, y) heatmap of
the RND intrinsic reward (predictor-vs-target error). The RND observation is the
agent position, so the reward is goal-independent and one map suffices.

These are live-only: `record` is accepted for signature parity with the n_dim
visualisers but ignored, and `save()` is a no-op.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt


# Goals are pinned to the right edge (x=1); these are the y positions shown.
GOAL_YS = (0.1, 0.5, 0.9)
GOAL_X = 1.0


def _draw_walls(ax, wall_x, hole_lo, hole_hi, color="cyan", lw=2):
    """Overlay each wall as two segments, leaving its hole open."""
    for wx, lo, hi in zip(wall_x, hole_lo, hole_hi):
        ax.plot([wx, wx], [0.0, lo], color=color, lw=lw, zorder=4)
        ax.plot([wx, wx], [hi, 1.0], color=color, lw=lw, zorder=4)


def _mark_goal(ax, gy):
    ax.plot(GOAL_X, gy, marker="*", color="gold", markersize=16,
            markeredgecolor="black", markeredgewidth=0.6, zorder=6)


class _GridMixin:
    """Builds and caches a flattened (res*res, 2) grid of (x, y) states."""

    def _build_grid(self, res):
        axis = torch.linspace(0.0, 1.0, res, device=self._device)
        y_grid, x_grid = torch.meshgrid(axis, axis, indexing="ij")
        x_flat = x_grid.reshape(-1)
        y_flat = y_grid.reshape(-1)
        # row index -> y, col index -> x once reshaped back to (res, res).
        self._states = torch.stack([x_flat, y_flat], dim=1)
        np_axis = np.linspace(0.0, 1.0, res)
        self._X, self._Y = np.meshgrid(np_axis, np_axis)

    def _goal_tensor(self, gy):
        n = self._states.shape[0]
        g = torch.tensor([GOAL_X, gy], dtype=torch.float32, device=self._device)
        return g.expand(n, 2)


class WallsPolicyVisualiser(_GridMixin):
    QUIVER_RES = 21
    MAGNIFY = 1.0

    def __init__(self, runner, env, update_every: int = 1, record: bool = False):
        self._runner = runner
        self._max_step_size = env.max_step_size
        self._update_every = update_every
        self._step = 0
        self._device = env.device
        self._wall_x = env.wall_x.cpu().numpy()
        self._hole_lo = env.hole_lo.cpu().numpy()
        self._hole_hi = env.hole_hi.cpu().numpy()
        self._build_grid(self.QUIVER_RES)
        self._setup_figure()

    def _setup_figure(self):
        plt.ion()
        self._fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        self._axes = list(np.atleast_1d(axes).ravel())
        try:
            self._fig.canvas.manager.set_window_title("Walls policy")
        except Exception:
            pass

        res = self.QUIVER_RES
        zero = np.zeros((res, res))
        self._quivers = []
        for ax, gy in zip(self._axes, GOAL_YS):
            q = ax.quiver(
                self._X, self._Y, zero, zero,
                angles="xy", scale_units="xy", scale=1.0, width=0.004,
            )
            self._quivers.append(q)
            _draw_walls(ax, self._wall_x, self._hole_lo, self._hole_hi)
            _mark_goal(ax, gy)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_aspect("equal")
            ax.set_title(f"goal y = {gy}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        self._fig.suptitle("Policy: deterministic action")
        self._fig.tight_layout()
        self._fig.show()

    @torch.no_grad()
    def _action(self, goals):
        obs = {"policy": self._states, "goal": {"desired_goal": goals}}
        return self._runner.get_deterministic_action(obs)

    def maybe_update(self):
        self._step += 1
        if self._step % self._update_every == 0:
            self.update()

    def update(self):
        res = self.QUIVER_RES
        for quiver, gy in zip(self._quivers, GOAL_YS):
            action = self._action(self._goal_tensor(gy)) * self._max_step_size
            action = (action * self.MAGNIFY).cpu().numpy()
            u = action[:, 0].reshape(res, res)
            v = action[:, 1].reshape(res, res)
            quiver.set_UVC(u, v)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def save(self, dir_path: str):
        pass


class _WallsBaseValueVisualiser(_GridMixin):
    """Value heatmap over (x, y) per pinned goal. Subclasses implement `_value`."""

    VALUE_RES = 50
    _title = "Value"

    def __init__(self, env, update_every: int = 2, record: bool = False):
        self._update_every = update_every
        self._step = 0
        self._device = env.device
        self._wall_x = env.wall_x.cpu().numpy()
        self._hole_lo = env.hole_lo.cpu().numpy()
        self._hole_hi = env.hole_hi.cpu().numpy()
        self._build_grid(self.VALUE_RES)
        self._setup_figure()

    def _setup_figure(self):
        plt.ion()
        self._fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        self._axes = list(np.atleast_1d(axes).ravel())
        try:
            self._fig.canvas.manager.set_window_title(self._title)
        except Exception:
            pass

        res = self.VALUE_RES
        empty = np.zeros((res, res))
        self._images = []
        for ax, gy in zip(self._axes, GOAL_YS):
            im = ax.imshow(
                empty, origin="lower", extent=[0, 1, 0, 1],
                aspect="equal", interpolation="nearest", cmap="viridis",
            )
            self._images.append(im)
            _draw_walls(ax, self._wall_x, self._hole_lo, self._hole_hi)
            _mark_goal(ax, gy)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"goal y = {gy}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        self._fig.colorbar(self._images[-1], ax=self._axes, label="value")
        self._fig.suptitle(self._title)
        self._fig.show()

    @torch.no_grad()
    def _value(self, states, goals):
        raise NotImplementedError

    def maybe_update(self):
        self._step += 1
        if self._step % self._update_every == 0:
            self.update()

    def update(self):
        res = self.VALUE_RES
        grids = []
        for gy in GOAL_YS:
            v = self._value(self._states, self._goal_tensor(gy)).cpu().numpy()
            grids.append(v.reshape(res, res))
        vmin = min(g.min() for g in grids)
        vmax = max(g.max() for g in grids)
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6
        for im, grid in zip(self._images, grids):
            im.set_data(grid)
            im.set_clim(vmin=vmin, vmax=vmax)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def save(self, dir_path: str):
        pass


class WallsIntrinsicRewardVisualiser(_GridMixin):
    """(x, y) heatmap of the RND intrinsic reward (predictor-vs-target error).

    The RND observation is the agent position, so the intrinsic reward is a pure
    function of (x, y) — goal-independent, so a single heatmap suffices. SAC-only
    and requires the runner to have RND enabled (use_rnd=True). The reward is the
    raw prediction error; the agent's reward normaliser only rescales it by a
    global std, which leaves the spatial pattern (and hence this map) unchanged.
    """

    REWARD_RES = 80
    _title = "RND intrinsic reward — predictor error"

    def __init__(self, runner, env, update_every: int = 2, record: bool = False):
        self._rnd = runner.runner.rnd
        if self._rnd is None:
            raise ValueError(
                "WallsIntrinsicRewardVisualiser requires RND (use_rnd=True)"
            )
        self._update_every = update_every
        self._step = 0
        self._device = env.device
        self._wall_x = env.wall_x.cpu().numpy()
        self._hole_lo = env.hole_lo.cpu().numpy()
        self._hole_hi = env.hole_hi.cpu().numpy()
        self._build_grid(self.REWARD_RES)
        self._setup_figure()

    def _setup_figure(self):
        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(6.5, 5.5))
        try:
            self._fig.canvas.manager.set_window_title("Walls intrinsic reward")
        except Exception:
            pass

        empty = np.zeros((self.REWARD_RES, self.REWARD_RES))
        self._im = self._ax.imshow(
            empty, origin="lower", extent=[0, 1, 0, 1],
            aspect="equal", interpolation="nearest", cmap="magma",
        )
        self._fig.colorbar(self._im, ax=self._ax, label="intrinsic reward")
        _draw_walls(self._ax, self._wall_x, self._hole_lo, self._hole_hi)
        self._ax.set_xlim(0.0, 1.0)
        self._ax.set_ylim(0.0, 1.0)
        self._ax.set_xlabel("x")
        self._ax.set_ylabel("y")
        self._fig.suptitle(self._title)
        self._fig.tight_layout()
        self._fig.show()

    @torch.no_grad()
    def _reward(self):
        return self._rnd.get_intrinsic_rew(self._states).squeeze(-1)

    def maybe_update(self):
        self._step += 1
        if self._step % self._update_every == 0:
            self.update()

    def update(self):
        res = self.REWARD_RES
        grid = self._reward().cpu().numpy().reshape(res, res)
        self._im.set_data(grid)
        vmin = float(grid.min())
        vmax = float(grid.max())
        if vmax - vmin < 1e-9:
            vmax = vmin + 1e-9
        self._im.set_clim(vmin=vmin, vmax=vmax)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def save(self, dir_path: str):
        pass


class WallsSACValueVisualiser(_WallsBaseValueVisualiser):
    NUM_ACTION_SAMPLES = 16
    _title = "SAC value — mean min(q1, q2) over sampled actions"

    def __init__(self, runner, env, update_every: int = 2, record: bool = False):
        self._sac = runner.runner
        super().__init__(env, update_every, record=record)

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


class WallsPPOValueVisualiser(_WallsBaseValueVisualiser):
    _title = "PPO value — v(s, g)"

    def __init__(self, runner, env, update_every: int = 2, record: bool = False):
        self._ppo = runner.runner
        super().__init__(env, update_every, record=record)

    @torch.no_grad()
    def _value(self, states, goals):
        ppo = self._ppo
        ppo.v.eval()
        obs = {"policy": states, "goal": {"desired_goal": goals}}
        return ppo.v(ppo.add_goal_obs(obs)).squeeze(-1)


class WallsBufferVisualiser:
    """(x, y) histograms of buffer transitions by state: mean reward and count."""

    NUM_BINS = 40
    NUM_SAMPLES = 15_000

    def __init__(self, runner, env, update_every: int = 20, record: bool = False):
        self._buffer = runner.runner.buffer
        self._update_every = update_every
        self._step = 0
        self._wall_x = env.wall_x.cpu().numpy()
        self._hole_lo = env.hole_lo.cpu().numpy()
        self._hole_hi = env.hole_hi.cpu().numpy()
        self._setup_figure()

    def _setup_figure(self):
        plt.ion()
        self._fig, (self._rew_ax, self._cnt_ax) = plt.subplots(1, 2, figsize=(13, 6))
        try:
            self._fig.canvas.manager.set_window_title("Walls buffer")
        except Exception:
            pass

        bins = self.NUM_BINS
        empty = np.full((bins, bins), np.nan)
        self._rew_im = self._rew_ax.imshow(
            empty, origin="lower", extent=[0, 1, 0, 1],
            aspect="equal", interpolation="nearest", cmap="RdYlGn",
            vmin=-0.01, vmax=1.0,
        )
        self._fig.colorbar(self._rew_im, ax=self._rew_ax, label="mean reward")
        self._cnt_im = self._cnt_ax.imshow(
            empty, origin="lower", extent=[0, 1, 0, 1],
            aspect="equal", interpolation="nearest", cmap="Blues",
        )
        self._fig.colorbar(self._cnt_im, ax=self._cnt_ax, label="transitions")

        for ax, title in ((self._rew_ax, "mean reward"), (self._cnt_ax, "transitions")):
            _draw_walls(ax, self._wall_x, self._hole_lo, self._hole_hi)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        self._fig.suptitle("Replay buffer: transitions by state position")
        self._fig.tight_layout()
        self._fig.show()

    def maybe_update(self):
        self._step += 1
        if self._step % self._update_every == 0:
            self.update()

    def update(self):
        buf = self._buffer
        filled = buf.capacity if buf.full else buf.idx
        if filled == 0:
            return

        n_samples = min(filled, self.NUM_SAMPLES)
        idx = torch.randint(0, filled, (n_samples,))
        obses = buf.obs[idx].cpu().numpy()
        rewards = buf.rew[idx].squeeze(-1).cpu().numpy()
        x = obses[:, 0]
        y = obses[:, 1]
        edges = np.linspace(0.0, 1.0, self.NUM_BINS + 1)

        counts, _, _ = np.histogram2d(x, y, bins=edges)
        reward_sum, _, _ = np.histogram2d(x, y, bins=edges, weights=rewards)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_reward = np.where(counts > 0, reward_sum / counts, np.nan)

        # histogram2d indexes [x_bin, y_bin]; imshow wants [y, x] -> transpose.
        self._rew_im.set_data(mean_reward.T)
        count_grid = np.where(counts > 0, counts, np.nan)
        self._cnt_im.set_data(count_grid.T)
        cmax = counts.max() if counts.max() > 0 else 1.0
        self._cnt_im.set_clim(vmin=0, vmax=cmax)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def save(self, dir_path: str):
        pass
