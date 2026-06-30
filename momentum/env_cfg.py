from dataclasses import dataclass
from walls.env_cfg import WallsVecEnvCfg

@dataclass    
class MomentumVecEnvCfg(WallsVecEnvCfg):
    momentum: float = 0.9
    n_walls: int = 1
    max_step_size: float = 0.01
    max_ep_len: int = 1000