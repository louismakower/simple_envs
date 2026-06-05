from louis_rl.sac import SACRunnerCfg
from n_dim.agents.her_cfg import NDimHERCfg

SAC_CFG = SACRunnerCfg(
    experiment_name="sac_nd",

    gamma=0.99,
    alpha_init=0.01,
    alpha_lr=3e-4,
    target_entropy="auto",

    replay_buffer_size=1_000_000,
    warmup_transitions=200_000,

    q_hidden_dims=[32, 32],
    q_learning_rate=3e-4,
    q_tau=0.005,
    q_grad_clip_norm=10.0,

    policy_hidden_dims=[32, 32],
    logstd_min=-5.0,
    logstd_max=2.0,
    policy_learning_rate=3e-4,

    reward_scaling=True,
    reward_G_max=5.0,
    reward_clip=10.0,
    reward_norm_type="ema",
    reward_ema_param=0.999,

    max_steps=8_000,
    steps_per_iter=1,
    num_train_updates=10,
    batch_size=1024,

    save_interval=600,

    her_cfg=NDimHERCfg(mode="future", k=4),
)
