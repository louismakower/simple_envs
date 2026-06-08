"""Live visualisations for the n-dim env.

`PolicyVisualiser` shows a quiver of the policy's first two action components
over a grid of (s_0, s_1). Other state and goal dims are pinned to 0.5. The
2x2 figure mirrors the two_dim convention: one subplot per fixed first-2-dim
goal corner from `GOALS`. For n=1 the figure collapses to a 1x2 grid and the
second state dimension is padded by tiling s_0.

`SACValueVisualiser` / `PPOValueVisualiser` show value vs. distance-to-goal as
a scatter over randomly sampled (state, goal) pairs in [0,1]^n.

`BufferVisualiser` shows, for each state dimension k, a 2D histogram of
(state_k, goal_k) pairs in the replay buffer, coloured by mean reward. One
subplot per dimension.
"""

import math
import os

import numpy as np
import torch
import matplotlib.pyplot as plt


# First-2-dim goal corners (permutations of (0.1, 0.9)). Other dims = 0.5.
GOALS = [(0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)]
# Goals when n == 1: just two positions along the single axis.
GOALS_1D = [0.1, 0.9]
# Pinned value for non-plotted state/goal dimensions (n >= 3).
FILL_VALUE = 0.5


class _Recorder:
    """Accumulates per-step snapshot dicts; writes one .npz on save()."""

    def __init__(self):
        self._snaps = []
        self._steps = []

    def add(self, step: int, snap):
        if snap is None:
            return
        self._snaps.append(snap)
        self._steps.append(step)

    def save(self, path: str, meta: dict):
        if not self._snaps:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = {}
        for key in self._snaps[0].keys():
            out[key] = np.stack([s[key] for s in self._snaps], axis=0)
        out["steps"] = np.array(self._steps, dtype=np.int64)
        out.update(meta)
        np.savez_compressed(path, **out)


class PolicyVisualiser:
    QUIVER_RES = 21
    MAGNIFY = 1.0
    SNAPSHOT_NAME = "policy.npz"

    def __init__(self, runner, env, update_every: int = 1, record: bool = False):
        self._runner = runner
        self._env = env
        self._n = env.n
        self._max_step_size = env.max_step_size
        self._update_every = update_every
        self._step = 0
        self._device = env.device
        self._recorder = _Recorder() if record else None
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

    def _compute(self):
        res = self.QUIVER_RES
        axis = torch.linspace(0.0, 1.0, res, device=self._device)
        y_grid, x_grid = torch.meshgrid(axis, axis, indexing="ij")
        x_flat = x_grid.reshape(-1)
        y_flat = y_grid.reshape(-1)
        states = self._build_states(x_flat, y_flat)

        uvs = []
        for goal in self._goals:
            goals = self._build_goals(goal, states.shape[0])
            action = self._action(states, goals) * self._max_step_size
            action = (action * self.MAGNIFY).cpu().numpy()
            if self._n == 1:
                u = action[:, 0].reshape(res, res)
                v = np.zeros_like(u)
            else:
                u = action[:, 0].reshape(res, res)
                v = action[:, 1].reshape(res, res)
            uvs.append(np.stack([u, v], axis=0))
        return {"uv": np.stack(uvs, axis=0).astype(np.float32)}

    def _draw(self, snap):
        for quiver, uv in zip(self._quivers, snap["uv"]):
            quiver.set_UVC(uv[0], uv[1])
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def update(self):
        snap = self._compute()
        if self._recorder is not None:
            self._recorder.add(self._step, snap)
        self._draw(snap)

    def save(self, dir_path: str):
        if self._recorder is None:
            return
        meta = {
            "kind": np.array("policy"),
            "n": np.int64(self._n),
            "max_step_size": np.float32(self._max_step_size),
            "quiver_res": np.int64(self.QUIVER_RES),
            "goals": np.array(self._goals, dtype=np.float32),
        }
        self._recorder.save(os.path.join(dir_path, self.SNAPSHOT_NAME), meta)


class _BaseValueVisualiser:
    """Value vs distance-to-goal scatter. Subclasses implement `_value`."""

    NUM_SAMPLES = 1000

    _title = "Value vs distance to goal"
    _ylabel = "V"
    _kind = "value"
    SNAPSHOT_NAME = "value.npz"

    def __init__(self, env, update_every: int = 20, record: bool = False):
        self._env = env
        self._n = env.n
        self._update_every = update_every
        self._step = 0
        self._device = env.device
        self._recorder = _Recorder() if record else None
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

    def _compute(self):
        states = torch.rand(self.NUM_SAMPLES, self._n, device=self._device)
        goals = torch.rand(self.NUM_SAMPLES, self._n, device=self._device)
        dists = torch.norm(states - goals, dim=-1).cpu().numpy()
        values = self._value(states, goals).cpu().numpy()
        return {
            "dists": dists.astype(np.float32),
            "values": values.astype(np.float32),
        }

    def _draw(self, snap):
        self._scatter.set_offsets(np.column_stack([snap["dists"], snap["values"]]))
        self._ax.set_ylim(float(snap["values"].min()), float(snap["values"].max()) + 1e-6)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def update(self):
        snap = self._compute()
        if self._recorder is not None:
            self._recorder.add(self._step, snap)
        self._draw(snap)

    def save(self, dir_path: str):
        if self._recorder is None:
            return
        meta = {
            "kind": np.array(self._kind),
            "n": np.int64(self._n),
            "title": np.array(self._title),
            "ylabel": np.array(self._ylabel),
        }
        self._recorder.save(os.path.join(dir_path, self.SNAPSHOT_NAME), meta)


class SACValueVisualiser(_BaseValueVisualiser):
    NUM_ACTION_SAMPLES = 16
    _title = "SAC value vs distance to goal"
    _ylabel = "mean min(q1, q2) over sampled actions"
    _kind = "sac_value"
    SNAPSHOT_NAME = "sac_value.npz"

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


class PPOValueVisualiser(_BaseValueVisualiser):
    _title = "PPO value vs distance to goal"
    _ylabel = "V"
    _kind = "ppo_value"
    SNAPSHOT_NAME = "ppo_value.npz"

    def __init__(self, runner, env, update_every: int = 2, record: bool = False):
        self._ppo = runner.runner
        super().__init__(env, update_every, record=record)

    @torch.no_grad()
    def _value(self, states, goals):
        ppo = self._ppo
        ppo.v.eval()
        obs = {"policy": states, "goal": {"desired_goal": goals}}
        return ppo.v(ppo.add_goal_obs(obs)).squeeze(-1)


class BufferVisualiser:
    """Per-dimension 2D histograms of (state_k, goal_k): mean reward and transition count."""

    NUM_BINS = 25
    NUM_SAMPLES = 15_000
    SNAPSHOT_NAME = "buffer.npz"

    def __init__(self, runner, env, update_every: int = 20, record: bool = False):
        self._buffer = runner.runner.buffer
        self._policy_obs_dim = runner.runner.policy_obs_dim
        self._n = env.n
        self._update_every = update_every
        self._step = 0
        self._recorder = _Recorder() if record else None
        self._setup_figure()

    def _setup_figure(self):
        plt.ion()
        n = self._n
        self._fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n), squeeze=False)
        try:
            self._fig.canvas.manager.set_window_title("Buffer: state vs goal")
        except Exception:
            pass

        bins = self.NUM_BINS
        empty = np.full((bins, bins), np.nan)
        self._reward_images = []
        self._count_images = []

        for k in range(n):
            rew_ax = axes[k, 0]
            cnt_ax = axes[k, 1]

            rim = rew_ax.imshow(
                empty,
                origin="lower", extent=[0, 1, 0, 1],
                aspect="equal", interpolation="nearest",
                cmap="RdYlGn",
            )
            self._fig.colorbar(rim, ax=rew_ax, label="mean reward")
            rew_ax.set_xlabel(f"state dim {k}")
            rew_ax.set_ylabel(f"goal dim {k}")
            rew_ax.set_title(f"dim {k} — mean reward")
            self._reward_images.append(rim)

            cim = cnt_ax.imshow(
                empty,
                origin="lower", extent=[0, 1, 0, 1],
                aspect="equal", interpolation="nearest",
                cmap="Blues",
            )
            self._fig.colorbar(cim, ax=cnt_ax, label="transitions")
            cnt_ax.set_xlabel(f"state dim {k}")
            cnt_ax.set_ylabel(f"goal dim {k}")
            cnt_ax.set_title(f"dim {k} — transitions")
            self._count_images.append(cim)

        self._fig.suptitle("Replay buffer: state vs goal")
        self._fig.tight_layout()
        self._fig.show()

    def maybe_update(self):
        self._step += 1
        if self._step % self._update_every == 0:
            self.update()

    def _compute(self):
        buf = self._buffer
        filled = buf.capacity if buf.full else buf.idx
        if filled == 0:
            return None

        n_samples = min(filled, self.NUM_SAMPLES)
        idx = torch.randint(0, filled, (n_samples,))
        obses = buf.obs[idx].cpu().numpy()
        rewards = buf.rew[idx].squeeze(-1).cpu().numpy()
        p = self._policy_obs_dim
        edges = np.linspace(0.0, 1.0, self.NUM_BINS + 1)

        mean_rewards = []
        counts_list = []
        for k in range(self._n):
            state_k = obses[:, k]
            goal_k = obses[:, p + k]
            counts, _, _ = np.histogram2d(state_k, goal_k, bins=edges)
            reward_sum, _, _ = np.histogram2d(state_k, goal_k, bins=edges, weights=rewards)
            mean_reward = np.where(counts > 0, reward_sum / counts, np.nan)
            mean_rewards.append(mean_reward)
            counts_list.append(counts)
        return {
            "mean_reward": np.stack(mean_rewards, axis=0).astype(np.float32),
            "counts": np.stack(counts_list, axis=0).astype(np.float32),
        }

    def _draw(self, snap):
        rew_vmin = -0.01
        rew_vmax = 1.
        for k, (rim, cim) in enumerate(zip(self._reward_images, self._count_images)):
            mean_reward = snap["mean_reward"][k]
            counts = snap["counts"][k]

            rim.set_data(mean_reward.T)
            rim.set_clim(vmin=rew_vmin, vmax=rew_vmax)

            count_grid = np.where(counts > 0, counts, np.nan)
            cim.set_data(count_grid.T)
            cmax = counts.max() if counts.max() > 0 else 1.0
            cim.set_clim(vmin=0, vmax=cmax)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def update(self):
        snap = self._compute()
        if self._recorder is not None:
            self._recorder.add(self._step, snap)
        if snap is not None:
            self._draw(snap)

    def save(self, dir_path: str):
        if self._recorder is None:
            return
        meta = {
            "kind": np.array("buffer"),
            "n": np.int64(self._n),
            "num_bins": np.int64(self.NUM_BINS),
        }
        self._recorder.save(os.path.join(dir_path, self.SNAPSHOT_NAME), meta)
