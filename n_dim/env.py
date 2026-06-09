import torch
from base_env import GoalReachVecEnv, reward_fn, _dist  # noqa: F401  (re-exported for HER)
from n_dim.env_cfg import NDimVecEnvCfg


class NDimVecEnv(GoalReachVecEnv):
    def __init__(
            self,
            cfg: NDimVecEnvCfg,
            device="cuda",
    ):
        self.n = cfg.n
        super().__init__(
            dim=cfg.n,
            num_envs=cfg.num_envs,
            max_ep_len=cfg.max_ep_len,
            goal_radius=cfg.goal_radius,
            max_step_size=cfg.max_step_size,
            device=device,
        )

    def _create_goals(self, num_goals=None):
        if num_goals is None:
            num_goals = self.num_envs
        return (torch.rand(size=(num_goals, self.dim), device=self._device) * 0.8) + 0.1

    def _sample_starts(self, num):
        return torch.rand(size=(num, self.dim), device=self._device)

    def _starts_valid(self, starts, goals):
        return _dist(starts, goals) >= self.goal_radius


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    num_envs = 10_0000
    ep_len = 300
    n = 1
    num_steps = 1 * ep_len

    env_cfg = NDimVecEnvCfg(num_envs=num_envs, max_ep_len=ep_len, n=1, max_step_size=0.1, goal_radius=0.1)
    env = NDimVecEnv(env_cfg, device="cuda")
    env.reset()

    all_steps = []
    all_positions = []

    for _ in range(num_steps):
        all_steps.append(env.ep_counters.cpu().numpy().copy())
        all_positions.append(env.state[:, 0].cpu().numpy().copy())
        action = torch.rand(num_envs, n, device=env.device) * 2 - 1
        env.step(action)

    steps = np.concatenate(all_steps)
    positions = np.concatenate(all_positions)

    counts, xedges, yedges = np.histogram2d(steps, positions, bins=[ep_len, 80], range=[[0, ep_len], [0, 1]])
    # Normalise each vertical slice (each step column) to a probability distribution over position.
    col_sums = counts.sum(axis=1, keepdims=True)
    counts = np.divide(counts, col_sums, out=np.zeros_like(counts), where=col_sums > 0)

    plt.figure(figsize=(12, 5))
    plt.pcolormesh(xedges, yedges, counts.T, cmap="hot", vmin=0, vmax=0.08)
    plt.colorbar(label="fraction of envs per step")
    plt.xlabel("steps since reset")
    plt.ylabel("position")
    plt.tight_layout()
    plt.savefig("fig.png")
