from louis_rl.algos.sac import SACRunnerCfg
from louis_rl.implementations.intrinsic import StableCountsCfg

SAC_CFG = SACRunnerCfg(
    experiment_name="sac_walls",

    gamma=0.99,
    alpha_init=0.2,
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

    max_steps=145_000,
    steps_per_iter=1,
    num_train_updates=10,
    batch_size=1024,

    save_interval=600,

    her_cfg=None,

    # default: no intrinsic reward; override per-run with --intrinsic. The
    # intrinsic_cfg must still construct even when disabled, so point it at the
    # counts config by default.
    use_intrinsic=True,
    intrinsic_cfg=StableCountsCfg(
        limits=[(0,1), (0,1)],
        resolutions=0.02,
        use_frac=0.25,
        stable_dims=[2, 3],  # rnd obs is [x, y, vx, vy] - filter by velocity
        stable_threshold=0.001,  # ~0.1 x max_step_size (0.01)
        ungated_weight=0.2,  # 0 = pure gated
    ),
    intrinsic_critic_hidden_layers=[128],
    intrinsic_critic_lr=3e-4,
    intrinsic_rew_clip=0.0,
    intrinsic_rew_weight=1.0e1,
    intrinsic_critic_tau=0.005,
    intrinsic_gamma=0.99,
    intrinsic_G_max=5.0,
)
