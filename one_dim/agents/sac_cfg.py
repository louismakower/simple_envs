from louis_rl.sac import SACRunnerCfg
from one_dim.agents.her_cfg import OneDimHERCfg

SAC_CFG = SACRunnerCfg(
    experiment_name = "sac_1d",

    gamma = 0.99,
    alpha_init = 0.2,
    alpha_lr = 3e-4,
    target_entropy = "auto",

    replay_buffer_size = 100_000,
    warmup_transitions = 10_000,

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

    max_steps = 200_000,
    steps_per_iter = 1,
    num_train_updates = 10,
    batch_size = 512,

    save_interval = 600,

    collect_states = False,

    her_cfg = OneDimHERCfg(mode="future")
)