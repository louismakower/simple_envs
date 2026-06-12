from louis_rl.sac import SACRunnerCfg
from louis_rl.intrinsic import RNDCfg
from walls.agents.her_cfg import NDimHERCfg

SAC_CFG = SACRunnerCfg(
    experiment_name="sac_walls",

    gamma=0.99,
    alpha_init=0.1,
    alpha_lr=3e-4,
    target_entropy="auto",

    replay_buffer_size=1_000_000,
    warmup_transitions=200_000,

    q_hidden_dims=[128, 128, 64],
    q_learning_rate=3e-4,
    q_tau=0.005,
    q_grad_clip_norm=10.0,

    policy_hidden_dims=[128, 128, 64],
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

    use_rnd = False,
    rnd_cfg=RNDCfg(
        pred_dim=10,
        target_hidden_layers=[10, 5],
        predictor_hidden_layers=[5],
        lr=3e-5,
        obs_clip=5.0,
        use_frac=0.25,
    ),
    rnd_critic_hidden_layers=[16],
    rnd_critic_lr=3e-6,
    rnd_rew_clip=0.0,
    rnd_rew_weight=1.0,
    rnd_critic_tau=0.005,
    rnd_gamma=0.99,
    rnd_G_max=5.0,
)
