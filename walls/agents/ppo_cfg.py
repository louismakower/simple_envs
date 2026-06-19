from louis_rl.algos.ppo import PPORunnerCfg
from louis_rl.implementations.intrinsic import CountsCfg, RNDCfg

# Canonical intrinsic-reward configs for the walls task, selected per-run by
# train.py's --intrinsic flag (none|counts|rnd). Kept here so the sweep and any
# single run share one definition of the hyperparameters.
PPO_COUNTS_CFG = CountsCfg(
    limits=[(0, 1), (0, 1)],
    resolutions=0.02,
    use_frac=0.25,
)
PPO_RND_CFG = RNDCfg(
    pred_dim=5,
    target_hidden_layers=[16],
    predictor_hidden_layers=[64, 64],
    lr=3e-4,
    obs_clip=5.0,
    use_frac=0.25,
)

PPO_CFG = PPORunnerCfg(
    experiment_name="ppo_walls",
    num_iterations=2000,
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

    # default: no intrinsic reward; override per-run with --intrinsic.
    intrinsic=None,
    intrinsic_gamma=0.99,
    intrinsic_v_grad_steps=10,
    intrinsic_V_hidden_layers=[128],
    intrinsic_V_lr=3e-4,
    intrinsic_weight=1.0e-1,
)
