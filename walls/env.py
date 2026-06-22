import torch
from base_env import GoalReachGoExploreVecEnv
from walls.env_cfg import WallsVecEnvCfg

# small offset so a blocked agent is placed just shy of the wall it hit
_EPS = 1e-4


class WallsVecEnv(GoalReachGoExploreVecEnv):
    def __init__(
            self,
            cfg: WallsVecEnvCfg,
            device="cpu",
    ):
        self.n_walls = cfg.n_walls
        self.gap_width = cfg.gap_width
        self._seed = cfg.seed
        self.cfg: WallsVecEnvCfg = cfg
        super().__init__(
            dim=2,
            num_envs=cfg.num_envs,
            max_ep_len=cfg.max_ep_len,
            goal_radius=cfg.goal_radius,
            max_step_size=cfg.max_step_size,
            goal_dynamics=cfg.goal_dynamics,
            device=device,
        )
        # walls + holes are sampled once and frozen, shared across all envs
        self.wall_x, self.hole_lo, self.hole_hi = self._create_walls(cfg.seed)

    def _create_goals(self, num_goals=None):
        # goal pinned to the right edge (x=1), uniform vertically
        if num_goals is None:
            num_goals = self.num_envs
        goals = torch.ones(size=(num_goals, 2), device=self._device)
        goals[:, 1] = torch.rand(size=(num_goals,), device=self._device)
        return goals

    def _sample_starts(self, num):
        # start pinned to the left edge (x=0)
        starts = torch.zeros(size=(num, 2), device=self._device)
        if not self.cfg.fixed_start:
            starts[:, 1] = torch.rand(size=(num,), device=self._device)
        return starts

    def _transition(self, state, scaled_action):
        next_state = torch.clamp(state + scaled_action, 0., 1.)
        return self._resolve_collisions(state, next_state)

    def _create_walls(self, seed):
        # evenly spaced interior vertical walls; one random hole per wall,
        # sampled once with a local generator so the maze is reproducible and
        # never touches the global / agent RNG.
        gen = torch.Generator(device=self._device).manual_seed(seed)
        wall_x = torch.arange(1, self.n_walls + 1, device=self._device) / (self.n_walls + 1)
        centers = (
            torch.rand(self.n_walls, generator=gen, device=self._device)
            * (1. - self.gap_width)
        ) + self.gap_width / 2
        hole_lo = centers - self.gap_width / 2
        hole_hi = centers + self.gap_width / 2
        return wall_x, hole_lo, hole_hi

    def _resolve_collisions(self, state, next_state):
        # slide against the nearest wall the straight move would cross outside a
        # hole: clamp x just shy of that wall, keep the full vertical move.
        x0 = state[:, 0:1]            # (N, 1)
        y0 = state[:, 1:2]
        x1 = next_state[:, 0:1]
        y1 = next_state[:, 1:2]
        wx = self.wall_x.unsqueeze(0)  # (1, W)

        dx = x1 - x0
        crossed = (x0 - wx) * (x1 - wx) <= 0  # (N, W) sign change, incl. landing exactly on wall

        safe_dx = torch.where(dx.abs() < 1e-12, torch.ones_like(dx), dx)
        t = (wx - x0) / safe_dx
        y_cross = y0 + t * (y1 - y0)
        in_hole = (y_cross >= self.hole_lo.unsqueeze(0)) & (y_cross <= self.hole_hi.unsqueeze(0))
        blocked = crossed & ~in_hole  # (N, W)

        dist = torch.where(blocked, (wx - x0).abs(), torch.full_like(crossed, float("inf"), dtype=x0.dtype))
        min_dist, min_idx = dist.min(dim=1)  # (N,)
        any_block = torch.isfinite(min_dist)

        block_wx = self.wall_x[min_idx]                       # (N,)
        sign_dx = torch.sign(dx.squeeze(1))                   # (N,)
        clamped_x = block_wx - _EPS * sign_dx

        out = next_state.clone()
        out[any_block, 0] = clamped_x[any_block]  # y already holds the full move
        return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    num_envs = 10_000
    ep_len = 200
    num_steps = 2 * ep_len

    env_cfg = WallsVecEnvCfg(num_envs=num_envs, max_ep_len=ep_len)
    env = WallsVecEnv(env_cfg, device="cpu")
    env.reset()

    all_positions = []
    for _ in range(num_steps):
        all_positions.append(env.state.cpu().numpy().copy())
        action = torch.rand(num_envs, 2, device=env.device) * 2 - 1
        env.step(action)

    positions = np.concatenate(all_positions)  # (M, 2)

    counts, xedges, yedges = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=[80, 80], range=[[0, 1], [0, 1]]
    )
    counts = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pcolormesh(xedges, yedges, counts.T, cmap="hot", vmin=0, vmax=np.quantile(counts, 0.999))

    # overlay walls (everything except the hole) so we can sanity-check collisions
    wall_x = env.wall_x.cpu().numpy()
    hole_lo = env.hole_lo.cpu().numpy()
    hole_hi = env.hole_hi.cpu().numpy()
    for wx, lo, hi in zip(wall_x, hole_lo, hole_hi):
        ax.plot([wx, wx], [0, lo], color="cyan", lw=2)
        ax.plot([wx, wx], [hi, 1], color="cyan", lw=2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("fig.png")
