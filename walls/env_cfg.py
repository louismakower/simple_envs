from dataclasses import dataclass

@dataclass
class WallsVecEnvCfg:
    num_envs: int = 10
    n_walls: int = 2
    gap_width: float = 0.1
    max_ep_len: int = 350
    goal_radius: float = 0.1
    max_step_size: float = 0.05
    seed: int = 42
