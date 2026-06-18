from louis_rl.algos.ppo import PPORunnerCfg
from louis_rl.implementations.intrinsic import CountsCfg, RNDCfg

PPO_CFG = PPORunnerCfg(
    experiment_name="ppo_walls",
    num_iterations=2000,
    steps_per_rollout=16,
    num_policy_grad_steps=10,
    num_v_grad_steps=10,
    policy_lr=3e-4,
    v_lr=3e-4,
    gamma=0.99,
    eps=0.2,
    save_interval=50,
    policy_hidden_dims=[256, 256, 64],
    v_hidden_dims=[64, 64],

    # intrinsic = None,
    intrinsic=CountsCfg(
        limits=[(0, 1), (0, 1)],
        resolutions=0.02,
    ),
    # intrinsic=RNDCfg(
    #     pred_dim=5,
    #     target_hidden_layers=[16],
    #     predictor_hidden_layers=[64, 64],
    #     lr=3e-4,
    #     obs_clip=0.0,
    #     use_frac=0.25,
    # ),
    intrinsic_gamma=0.999,
    intrinsic_v_grad_steps=10,
    intrinsic_V_hidden_layers=[256, 256, 256, 128],
    intrinsic_V_lr=3e-4,
    intrinsic_weight=1.0e-1,
)
