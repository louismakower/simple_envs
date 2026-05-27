from dataclasses import dataclass

@dataclass
class NDimVecEnvCfg:
    num_envs: int = 10
    n: int = 2
    max_ep_len: int = 100
    goal_radius: float = 0.1
    max_step_size: float = 0.1
