from dataclasses import dataclass

@dataclass
class WallsVecEnvCfg:
    num_envs: int = 10
    n_walls: int = 3
    gap_width: float = 0.1
    max_ep_len: int = 200
    goal_radius: float = 0.05
    max_step_size: float = 0.04
    seed: int = 42
