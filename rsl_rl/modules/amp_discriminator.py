"""AMP discriminator for adversarial motion priors.

Ported from humanoid_skateboarding DiscriminatorMulti with standardized naming.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from torch import autograd


class AMPDiscriminator(nn.Module):
  """Multi-frame discriminator with spectral normalization for AMP style reward."""

  def __init__(
    self,
    amp_obs_dim: int,
    amp_reward_coef: float = 2.0,
    hidden_layer_sizes: tuple[int, ...] = (256, 256),
    device: str = "cpu",
    amp_num_frames: int = 2,
    task_reward_lerp: float = 0.0,
    use_lerp: bool = True,
  ):
    super().__init__()
    self.device = device
    self.amp_obs_dim = amp_obs_dim
    self.use_lerp = use_lerp
    self.amp_num_frames = amp_num_frames
    self.amp_reward_coef = amp_reward_coef
    self.task_reward_lerp = task_reward_lerp

    # Build trunk: spectral_norm(Linear) + ReLU layers
    amp_layers: list[nn.Module] = []
    curr_in_dim = amp_obs_dim * amp_num_frames
    for hidden_dim in hidden_layer_sizes:
      amp_layers.append(spectral_norm(nn.Linear(curr_in_dim, hidden_dim)))
      amp_layers.append(nn.ReLU())
      curr_in_dim = hidden_dim
    self.trunk = nn.Sequential(*amp_layers).to(device)
    self.amp_linear = spectral_norm(nn.Linear(hidden_layer_sizes[-1], 1)).to(device)

    self.trunk.train()
    self.amp_linear.train()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass. x: (batch, amp_obs_dim * amp_num_frames) already flattened."""
    h = self.trunk(x)
    d = self.amp_linear(h)
    return d

  def compute_grad_pen(self, expert_states: torch.Tensor, lambda_: float = 10.0) -> torch.Tensor:
    """Gradient penalty on expert states. Target norm is 0.

    Args:
      expert_states: (batch, amp_num_frames, amp_obs_dim)
      lambda_: gradient penalty coefficient
    """
    expert_data = expert_states.flatten(1)
    expert_data.requires_grad = True

    disc = self.amp_linear(self.trunk(expert_data))
    ones = torch.ones(disc.size(), device=disc.device)
    grad = autograd.grad(
      outputs=disc,
      inputs=expert_data,
      grad_outputs=ones,
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]

    grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
    return grad_pen

  def predict_amp_reward(
    self,
    states: torch.Tensor,
    task_reward: torch.Tensor,
    normalizer=None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict AMP reward blended with task reward.

    Args:
      states: (num_envs, amp_num_frames, amp_obs_dim)
      task_reward: (num_envs,)
      normalizer: AMPNormalizer for state normalization

    Returns:
      (blended_reward, logit, disc_reward) all (num_envs,)
    """
    with torch.no_grad():
      self.eval()
      if normalizer is not None:
        states = normalizer.normalize_torch(states, self.device)

      state_cat = states.flatten(1)
      d = self.amp_linear(self.trunk(state_cat))
      disc_reward = self.amp_reward_coef * torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)

      if self.use_lerp:
        if self.task_reward_lerp > 0:
          reward = (1.0 - self.task_reward_lerp) * disc_reward + self.task_reward_lerp * task_reward.unsqueeze(-1)
        else:
          reward = disc_reward
        self.train()
        return reward.squeeze(-1), d, disc_reward.squeeze(-1) * (1.0 - self.task_reward_lerp)
      else:
        disc_reward *= 0.02
        reward = task_reward.unsqueeze(-1) + disc_reward
        self.train()
        return reward.squeeze(-1), d, disc_reward.squeeze(-1)

  def get_disc_weights(self) -> list[torch.Tensor]:
    weights = []
    for m in self.trunk.modules():
      if isinstance(m, nn.Linear):
        weights.append(torch.flatten(m.weight))
    weights.append(torch.flatten(self.amp_linear.weight))
    return weights

  def get_disc_logit_weights(self) -> torch.Tensor:
    return torch.flatten(self.amp_linear.weight)
