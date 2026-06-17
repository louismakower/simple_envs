from louis_rl.ppo import PPORunnerCfg
from louis_rl.intrinsic import RNDCfg

PPO_CFG = PPORunnerCfg(
    experiment_name="ppo_nd",
    num_iterations=2000,
    steps_per_rollout=16,
    num_policy_grad_steps=10,
    num_v_grad_steps=10,
    policy_lr=3e-4,
    v_lr=3e-4,
    gamma=0.99,
    eps=0.2,
    save_interval=50,
    policy_hidden_dims=[64, 64],
    v_hidden_dims=[64, 64],

    # rnd=None,
    intrinsic=RNDCfg(
        pred_dim=5,
        target_hidden_layers=[10],
        predictor_hidden_layers=[15],
        lr=3e-4,
        obs_clip=0.0,
        use_frac=0.25
    ),
    intrinsic_gamma=0.999,
    intrinsic_v_grad_steps=10,
    intrinsic_V_hidden_layers=[128, 128],
    intrinsic_V_lr=3e-4,
    intrinsic_weight=1.0
)
