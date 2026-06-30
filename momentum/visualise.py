"""Live spatial visualisations for the momentum walls env.

The momentum env is a `WallsVecEnv` whose policy observation is `[position,
velocity]`, so the policy/value visualisers are the walls ones with a velocity
block appended to each grid position. We visualise the **zero-velocity slice**
(the policy/value assuming the agent currently has no momentum), which keeps the
figures laid out identically to the walls env. Position and velocity share the
same width, so a `zeros_like` block suffices.

The intrinsic-reward and buffer visualisers need no changes: the RND obs is
position-only and the buffer obs is `[position, velocity, goal]`, so both already
read position straight off. They are re-exported here so `train.py` can pull the
whole momentum set from one module.
"""

import torch

from walls.visualise import (
    WallsPolicyVisualiser,
    WallsSACValueVisualiser,
    WallsPPOValueVisualiser,
    WallsPPOIntrinsicValueVisualiser,
    WallsBufferVisualiser,
    WallsIntrinsicRewardVisualiser,
)


class _ZeroVelMixin:
    """Append a zero-velocity block so grid positions match the [pos, vel] obs."""

    def _policy_obs(self, states):
        return torch.cat([states, torch.zeros_like(states)], dim=-1)


class MomentumPolicyVisualiser(_ZeroVelMixin, WallsPolicyVisualiser):
    pass


class MomentumSACValueVisualiser(_ZeroVelMixin, WallsSACValueVisualiser):
    pass


class MomentumPPOValueVisualiser(_ZeroVelMixin, WallsPPOValueVisualiser):
    pass


class MomentumPPOIntrinsicValueVisualiser(_ZeroVelMixin, WallsPPOIntrinsicValueVisualiser):
    pass


# reused unchanged — re-exported for a single import site in train.py
MomentumBufferVisualiser = WallsBufferVisualiser
MomentumIntrinsicRewardVisualiser = WallsIntrinsicRewardVisualiser
