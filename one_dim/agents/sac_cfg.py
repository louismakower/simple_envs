from louis_rl.sac import SACRunnerCfg

SAC_CFG = SACRunnerCfg(
    experiment_name = "sac_1d",

    gamma = 0.97,
    alpha_init = 0.2,
    alpha_lr = 3e-4,
    target_entropy = "auto",

    replay_buffer_size = 1_000_00,
    warmup_transitions = 1_0000,

    q_hidden_dims = [6, 6],
    q_learning_rate = 3e-4,
    q_tau = 0.005,
    q_grad_clip_norm = 10.0,

    policy_hidden_dims = [6, 6],
    logstd_min = -5.0,
    logstd_max = 2.0,
    policy_learning_rate = 3e-4,

    reward_scaling = True,
    reward_G_max = 5.0,
    reward_clip = 0.0,

    max_steps = 500_000,
    steps_per_iter = 1,
    num_train_updates = 1,
    batch_size = 256,

    save_interval = 600,

    collect_states = False,

    # her_cfg = FrankaReachHerCfg(mode="future", success_threshold=0.03)
)