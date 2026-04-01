"""Actor-critic with per-cost softplus value heads for P3O constrained RL."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic


class ActorCriticWithCostHeads(ActorCritic):
  """Extends ActorCritic with independent cost value heads for P3O.

  Each cost channel gets its own MLP value head with softplus output
  (non-negative cost value prediction). The cost heads share the critic
  observation group but have independent parameters.
  """

  def __init__(
    self,
    obs,
    obs_groups,
    num_actions,
    num_costs: int = 0,
    cost_hidden_dims: tuple[int, ...] = (256, 256),
    **kwargs,
  ):
    super().__init__(obs, obs_groups, num_actions, **kwargs)
    self.num_costs = num_costs
    if num_costs > 0:
      # Infer critic obs dim from the critic MLP's input size
      critic_input_dim = self.critic[0].in_features
      self.cost_value_heads = nn.ModuleList(
        [self._build_cost_mlp(critic_input_dim, cost_hidden_dims) for _ in range(num_costs)]
      )

  def _build_cost_mlp(self, input_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    curr_dim = input_dim
    for h in hidden_dims:
      layers += [nn.Linear(curr_dim, h), nn.ELU()]
      curr_dim = h
    layers += [nn.Linear(curr_dim, 1), nn.Softplus()]
    return nn.Sequential(*layers)

  def evaluate_costs(self, obs, **kwargs) -> torch.Tensor:
    """Predict cost values for all cost channels.

    Args:
      obs: Observation TensorDict (uses critic observation group).

    Returns:
      (batch, num_costs) tensor of non-negative cost value predictions.
    """
    if self.num_costs == 0:
      batch_size = next(iter(obs.values())).shape[0] if hasattr(obs, "values") else obs.shape[0]
      return torch.zeros(batch_size, 0, device=next(self.parameters()).device)
    # Use parent's get_critic_obs + normalizer (same input as reward critic)
    critic_obs = self.get_critic_obs(obs)
    critic_obs = self.critic_obs_normalizer(critic_obs)
    return torch.cat([head(critic_obs) for head in self.cost_value_heads], dim=-1)
