import torch
from louis_rl.vec_env import SpaceInfo
from n_dim.env_cfg import NDimVecEnvCfg

class NDimVecEnv:
    def __init__(
            self,
            cfg: NDimVecEnvCfg,
            device="cpu",
    ):
        self.n = cfg.n
        self._num_envs = cfg.num_envs
        self.max_ep_len = cfg.max_ep_len
        self.goal_radius = cfg.goal_radius
        self.max_step_size = cfg.max_step_size
        self._device = device

        self.state = torch.rand(
            size=(self.num_envs, self.n)
        )
        self.goal = torch.rand(
            size=(self.num_envs, self.n)
        )

        self.ep_counters = torch.zeros(size=(self.num_envs,))

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
        }
    
    def step(self, action):
        clipped_action = torch.clip(action, -1., 1.)
        scaled_action = clipped_action * self.max_step_size

        next_state = self.state + scaled_action

        # clip to environment bounds
        next_state = torch.clamp(next_state, 0., 1.)
        self.ep_counters += 1

        rew = self.compute_rew(next_state)
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
        extras = {"terminal_obs": terminal_obs}

        for visualiser in self._visualisers:
            visualiser.maybe_update()

        return obs, rew, term, timeout, extras
    
    def get_obs(self, state, goal):
        return {
            "policy": state.clone(),
            "goal": {"desired_goal": goal.clone()},
            "her": {"position": state.clone()},
        }
    
    def get_terminal_obs(self, state, goal, env_ids):
        nan = float("nan")
        terminal_state = torch.where(env_ids.unsqueeze(-1), state, nan)
        terminal_goal = torch.where(env_ids.unsqueeze(-1), goal, nan)
        return self.get_obs(terminal_state, terminal_goal)
    

    def reset(self, env_ids: torch.Tensor = None):
        # reset all if no env ids are given
        if env_ids is None:
            env_ids = torch.ones(size=(self.num_envs,), dtype=torch.bool)

        num_reset_envs = torch.sum(env_ids).int()
        new_goals = torch.rand(size=(num_reset_envs, self.n))
        self.goal[env_ids] = new_goals

        new_start_pos = torch.rand(size=(num_reset_envs, self.n))
        far_enough = self._dist(new_start_pos, new_goals) >= self.goal_radius
        while not far_enough.all():
            new_start_pos[~far_enough] = torch.rand(size=((~far_enough).sum(), self.n))
            far_enough = self._dist(new_start_pos, new_goals) >= self.goal_radius


        self.state[env_ids] = new_start_pos
        self.ep_counters[env_ids] = 0

        return self.get_obs(self.state, self.goal), {}


    def _dist(self, state, goal):
        return torch.norm(
            state - goal,
            dim=-1,
        )
    
    def compute_rew(self, state):
        reached = (self._dist(state, self.goal) < self.goal_radius).float() * self.n
        neg_step = -0.01
        return 10 * reached + neg_step
    
    def compute_term(self, state):
        return (self._dist(state, self.goal) < self.goal_radius)
    
    def compute_timeout(self):
        return self.ep_counters >= self.max_ep_len
    

if __name__ == "__main__":

    env_cfg = NDimVecEnvCfg()
    env = NDimVecEnv(env_cfg)
    env.reset()
    print(env.state, env.goal)
    while True:
        num = input("enter a step: ")
        action = torch.tensor([float(num)]).expand_as(env.state)
        obs, rew, term, timeout, extras = env.step(action)
        print("rew", rew)
        print("obs", obs)
        print("term", term)
        print("timeout", timeout)
        print(extras)
        print(env.state, env.goal)