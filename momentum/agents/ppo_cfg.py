from louis_rl.algos.ppo import PPORunnerCfg
from louis_rl.implementations.intrinsic import StableCountsCfg

PPO_CFG = PPORunnerCfg(
    experiment_name="ppo_walls",
    num_iterations=6000,
    steps_per_rollout=32,
    num_policy_grad_steps=10,
    num_v_grad_steps=10,
    policy_lr=3e-4,
    v_lr=3e-4,
    gamma=0.99,
    eps=0.2,
    save_interval=300,
    policy_hidden_dims=[128, 64],
    v_hidden_dims=[128, 64],

    intrinsic=StableCountsCfg(
        limits=[(0,1), (0,1)],
        resolutions=0.02,
        use_frac=0.25,
        stable_dims=[2, 3],  # rnd obs is [x, y, vx, vy] - filter by velocity
        stable_threshold=0.001,  # ~0.1 x max_step_size (0.01)
        ungated_weight=0.3,  # 0 = pure gated
    ),
    intrinsic_gamma=0.99,
    intrinsic_v_grad_steps=10,
    intrinsic_V_hidden_layers=[128],
    intrinsic_V_lr=3e-4,
    intrinsic_weight=1.0e-1,
)
