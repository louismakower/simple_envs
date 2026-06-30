# simple_envs

Lightweight vectorised goal-reaching environments for RL research, built on top of `louis_rl`.

A point agent moves through the unit hypercube toward a randomly sampled goal, receiving a sparse reward on success. Three environments are provided:

- **n_dim** — unconstrained navigation in an N-dimensional unit hypercube
- **walls** — 2-D navigation with vertical walls and gaps (frozen maze layout)
- **momentum** — the walls maze with inertia: each step blends the new action with the previous one, so the agent can't turn or stop instantly

All three support SAC and PPO, optionally with Hindsight Experience Replay (HER) and intrinsic motivation.

## Usage

```bash
python train.py --task n_dim --agent sac --n 2
python train.py --task walls --agent ppo --n_walls 3
python train.py --task momentum --agent sac
```

### Common arguments

| Argument | Description |
|---|---|
| `--task` | `n_dim`, `walls`, or `momentum` |
| `--agent` | `ppo` or `sac` |
| `--num_envs` | Number of parallel environments |
| `--max_ep_len` | Episode length |
| `--goal_radius` | Success radius |
| `--seed` | Global RNG seed |
| `--device` | `cuda` or `cpu` |
| `--no_visualise` | Disable live visualisation |
| `--record` | Save visualiser snapshots to `<log_dir>/snapshots` |

### n_dim-specific

| Argument | Description |
|---|---|
| `--n` | State/action dimensionality |

### walls-specific

| Argument | Description |
|---|---|
| `--n_walls` | Number of vertical walls |
| `--gap_width` | Gap height in each wall |
| `--maze_seed` | Seed for the frozen maze layout |

### momentum-specific

Momentum reuses the walls maze and has no dedicated CLI flags. The inertia coefficient is set in `momentum/env_cfg.py`
