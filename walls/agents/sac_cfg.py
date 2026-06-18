from louis_rl.algos.sac import SACRunnerCfg
from louis_rl.implementations.intrinsic import RNDCfg, CountsCfg
from walls.agents.her_cfg import NDimHERCfg

SAC_CFG = SACRunnerCfg(
    experiment_name="sac_walls",

    gamma=0.99,
    alpha_init=0.01,
    alpha_lr=3e-4,
    target_entropy="auto",

    replay_buffer_size=1_000_000,
    warmup_transitions=200_00,

    q_hidden_dims=[128, 64],
    q_learning_rate=3e-4,
    q_tau=0.005,
    q_grad_clip_norm=10.0,

    policy_hidden_dims=[128, 64],
    logstd_min=-5.0,
    logstd_max=2.0,
    policy_learning_rate=3e-4,

    reward_scaling=True,
    reward_G_max=5.0,
    reward_clip=10.0,
    reward_norm_type="ema",
    reward_ema_param=0.999,

    max_steps=3_000,
    steps_per_iter=1,
    num_train_updates=10,
    batch_size=1024,

    save_interval=600,

    # her_cfg=NDimHERCfg(mode="future", k=4),

    use_intrinsic = True,
    intrinsic_cfg=CountsCfg(
        limits=[(0, 1), (0, 1)],
        resolutions=0.02,
    ),
    # intrinsic_cfg=RNDCfg(
    #     pred_dim=10,
    #     target_hidden_layers=[16],
    #     predictor_hidden_layers=[64, 16],
    #     lr=3e-6,
    #     obs_clip=5.0,
    #     use_frac=0.25,
    # ),
    intrinsic_critic_hidden_layers=[128],
    intrinsic_critic_lr=3e-4,
    intrinsic_rew_clip=0.0,
    intrinsic_rew_weight=1.0,
    intrinsic_critic_tau=0.005,
    intrinsic_gamma=0.99,
    intrinsic_G_max=5.0,
)
