"""Live spatial visualisations for the momentum walls env.

The momentum env is a `WallsVecEnv` whose policy observation is `[position,
velocity]`, so the policy/value visualisers are the walls ones with a velocity
block appended to each grid position. We visualise the **zero-velocity slice**
(the policy/value assuming the agent currently has no momentum), which keeps the
figures laid out identically to the walls env. Position and velocity share the
same width, so a `zeros_like` block suffices.

The intrinsic obs is now `[position, velocity]`, so the intrinsic-reward visualiser
feeds each grid position with zero velocity (the reward for stopping there). The
buffer obs is `[position, velocity, goal]` and reads position straight off, so it
is re-exported unchanged, alongside the whole momentum set for one import in train.py.
"""

import torch

from walls.visualise import (
    WallsPolicyVisualiser,
    WallsSACValueVisualiser,
    WallsPPOValueVisualiser,
    WallsPPOIntrinsicValueVisualiser,
    WallsBufferVisualiser,
    WallsIntrinsicRewardVisualiser,
    WallsStableCountsVisualiser,
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


# reused unchanged — re-exported for a single import site in train.py. The
# stable-counts visualiser reads the intrinsic module's own grids (which are over
# position only), so it needs no velocity handling and works as-is here.
MomentumBufferVisualiser = WallsBufferVisualiser
MomentumStableCountsVisualiser = WallsStableCountsVisualiser


class MomentumIntrinsicRewardVisualiser(WallsIntrinsicRewardVisualiser):
    """Append zero velocity to each grid position, so the map shows the reward for
    coming to rest in that cell (bright = novel stable cell still to be claimed)."""

    @torch.no_grad()
    def _reward(self):
        pad = self._intrinsic.obs_dim - self._states.shape[1]
        obs = torch.cat([self._states, torch.zeros(self._states.shape[0], pad, device=self._states.device)], dim=-1)
        return self._intrinsic.get_intrinsic_rew(obs).squeeze(-1)
