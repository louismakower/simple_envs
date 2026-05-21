from louis_rl.ppo import PPORunnerCfg

PPO_CFG = PPORunnerCfg(
        experiment_name="ppo_1d",
        num_iterations=2000,
        steps_per_rollout=64,
        num_policy_grad_steps=10,
        num_v_grad_steps=10,
        policy_lr=3e-4,
        v_lr=3e-4,
        gamma=0.99,
        eps=0.2,
        save_interval=200,
        policy_hidden_dims=[16],
        v_hidden_dims=[16],
    )