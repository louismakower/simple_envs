from louis_rl.algos.sac import SACRunnerCfg
from louis_rl.implementations.intrinsic import RNDCfg, CountsCfg
from walls.agents.her_cfg import NDimHERCfg

# Canonical intrinsic-reward configs for the walls task, selected per-run by
# train.py's --intrinsic flag (none|counts|rnd). Kept here so the sweep and any
# single run share one definition of the hyperparameters.
SAC_COUNTS_CFG = CountsCfg(
    limits=[(0, 1), (0, 1)],
    resolutions=0.02,
    use_frac=0.25,
)
SAC_RND_CFG = RNDCfg(
    pred_dim=5,
    target_hidden_layers=[16],
    predictor_hidden_layers=[64, 64],
    lr=3e-4,
    obs_clip=5.0,
    use_frac=0.25,
)

SAC_CFG = SACRunnerCfg(
    experiment_name="sac_walls",

    gamma=0.99,
    alpha_init=0.1,
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

    max_steps=15_000,
    steps_per_iter=1,
    num_train_updates=10,
    batch_size=1024,

    save_interval=600,

    her_cfg=NDimHERCfg(mode="future", k=4),

    # default: no intrinsic reward; override per-run with --intrinsic. The
    # intrinsic_cfg must still construct even when disabled, so point it at the
    # counts config by default.
    use_intrinsic=False,
    intrinsic_cfg=SAC_COUNTS_CFG,
    intrinsic_critic_hidden_layers=[128],
    intrinsic_critic_lr=3e-4,
    intrinsic_rew_clip=0.0,
    intrinsic_rew_weight=1.0e-1,
    intrinsic_critic_tau=0.005,
    intrinsic_gamma=0.99,
    intrinsic_G_max=5.0,
)
