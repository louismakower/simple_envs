import torch
from louis_rl.vec_env import SpaceInfo
from n_dim.env_cfg import NDimVecEnvCfg

class NDimVecEnv:
    def __init__(
            self,
            cfg: NDimVecEnvCfg,
            device="cuda",
    ):
        self.n = cfg.n
        self._num_envs = cfg.num_envs
        self.max_ep_len = cfg.max_ep_len
        self.goal_radius = cfg.goal_radius
        self.max_step_size = cfg.max_step_size
        self._device = device

        self.state = torch.rand(size=(self.num_envs, self.n), device=self._device)
        self.goal = self._create_goals()

        self.ep_counters = torch.zeros(size=(self.num_envs,), device=self._device)

        self._visualisers = []

    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def num_envs(self) -> int:
        return self._num_envs
    
    @property
    def max_episode_length(self) -> int:
        return self.max_ep_len

    @property
    def action_space(self) -> SpaceInfo:
        return SpaceInfo(shape=(self.n,))
    
    @property
    def observation_space(self) -> dict:
        return {
            "policy": SpaceInfo(shape=(self.n,)),
            "goal": {"desired_goal": SpaceInfo(shape=(self.n,))},
            "rnd": SpaceInfo(shape=(self.n,))
        }
    
    def step(self, action):
        clipped_action = torch.clip(action, -1., 1.)
        scaled_action = clipped_action * self.max_step_size

        next_state = self.state + scaled_action

        # clip to environment bounds
        next_state = torch.clamp(next_state, 0., 1.)
        self.ep_counters += 1

        rew = self.compute_rew(next_state, self.goal)
        dist_to_goal = _dist(next_state, self.goal).mean()
        term = self.compute_term(next_state)
        timeout = self.compute_timeout()

        to_reset = term | timeout
        terminal_obs = self.get_terminal_obs(
            next_state,
            self.goal,
            env_ids=to_reset,
        )
        
        self.state[~to_reset] = next_state[~to_reset]
        
        # reset fn returns all obs
        obs, _ = self.reset(env_ids=to_reset)
        extras = {
            "terminal_obs": terminal_obs,
            "log": {"metrics/dist_to_goal": dist_to_goal},
            }

        for visualiser in self._visualisers:
            visualiser.maybe_update()

        return obs, rew, term, timeout, extras
    
    def get_obs(self, state, goal):
        return {
            "policy": state.clone(),
            "goal": {"desired_goal": goal.clone()},
            "her": {"position": state.clone()},
            "rnd": state.clone(),
        }
    
    def get_terminal_obs(self, state, goal, env_ids):
        nan = float("nan")
        terminal_state = torch.where(env_ids.unsqueeze(-1), state, nan)
        terminal_goal = torch.where(env_ids.unsqueeze(-1), goal, nan)
        return self.get_obs(terminal_state, terminal_goal)
    

    def reset(self, env_ids: torch.Tensor = None):
        # reset all if no env ids are given
        if env_ids is None:
            env_ids = torch.ones(size=(self.num_envs,), dtype=torch.bool, device=self._device)

        num_reset_envs = torch.sum(env_ids).int()
        new_goals = self._create_goals(num_reset_envs)
        self.goal[env_ids] = new_goals

        new_start_pos = torch.rand(size=(num_reset_envs, self.n), device=self._device)
        far_enough = _dist(new_start_pos, new_goals) >= self.goal_radius
        while not far_enough.all():
            new_start_pos[~far_enough] = torch.rand(size=((~far_enough).sum(), self.n), device=self._device)
            far_enough = _dist(new_start_pos, new_goals) >= self.goal_radius


        self.state[env_ids] = new_start_pos
        self.ep_counters[env_ids] = 0

        return self.get_obs(self.state, self.goal), {}

    def compute_rew(self, state, goal):
        return reward_fn(state, goal, self.goal_radius, self.n)
    
    def compute_term(self, state):
        return (_dist(state, self.goal) < self.goal_radius)
    
    def compute_timeout(self):
        return self.ep_counters >= self.max_ep_len
    
    def randomise_ep_counters(self):
        self.ep_counters = (torch.rand_like(self.ep_counters) * self.max_ep_len).long()
    
    def _create_goals(self, num_goals=None):
        if num_goals is None:
            num_goals = self.num_envs
        return (torch.rand(size=(num_goals, self.n), device=self._device) * 0.8) + 0.1

    
def reward_fn(state, goal, goal_radius, n):
    reached = (_dist(state, goal) < goal_radius).float()
    neg_step = -0.01
    return reached + neg_step

def _dist(pos1, pos2):
    return torch.norm(
        pos1 - pos2,
        dim=-1,
    )

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