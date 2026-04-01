"""P3O cost constraint mixin providing L_viol penalty and adaptive κ.

Implements P3O paper (2205.11814) Eq. 5-7 and Algorithm 2 adaptive κ scheduling.
This is a plain mixin — no __init__ chain. Composed classes must call
_init_cost_constraint() in their __init__ after the base class init.
"""

from __future__ import annotations

import torch


class CostConstraintMixin:
  """Mixin adding P3O cost penalty to any PPO-based algorithm.

  Must be mixed with a PPO subclass that provides:
  - self.device
  - self.storage (with cost buffers)
  - self.policy (with evaluate_costs method)
  """

  num_costs: int
  c_gamma: torch.Tensor  # (num_costs,)
  c_scale: torch.Tensor  # (num_costs,) — κ per cost
  cost_value_loss_coef: float
  adaptive_kappa: bool
  kappa_rho: float
  kappa_max: float
  normalize_cost: bool

  def _init_cost_constraint(
    self,
    num_costs: int,
    c_gamma: list[float] | torch.Tensor,
    c_scale: list[float] | torch.Tensor,
    cost_value_loss_coef: float = 1.0,
    adaptive_kappa: bool = True,
    kappa_rho: float = 1.5,
    kappa_max: float = 100.0,
    normalize_cost: bool = True,
  ):
    """Initialize cost constraint state. Call in __init__ after super().__init__()."""
    self.num_costs = num_costs
    if isinstance(c_gamma, torch.Tensor):
      self.c_gamma = c_gamma.to(self.device)
    else:
      self.c_gamma = torch.tensor(c_gamma, dtype=torch.float32, device=self.device)
    if isinstance(c_scale, torch.Tensor):
      self.c_scale = c_scale.to(self.device)
    else:
      self.c_scale = torch.tensor(c_scale, dtype=torch.float32, device=self.device)
    self.cost_value_loss_coef = cost_value_loss_coef
    self.adaptive_kappa = adaptive_kappa
    self.kappa_rho = kappa_rho
    self.kappa_max = kappa_max
    self.normalize_cost = normalize_cost

  def _compute_cost_loss(
    self,
    ratio: torch.Tensor,
    clipped_ratio: torch.Tensor,
    adv_c_batch: torch.Tensor,
    returns_c_batch: torch.Tensor,
    adv_c_mean: torch.Tensor,
    adv_c_std: torch.Tensor,
    cost_limits: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """P3O violation penalty (Eq. 5-7).

    Args:
      ratio: (batch,) importance sampling ratio r(θ).
      clipped_ratio: (batch,) clipped ratio.
      adv_c_batch: (batch, num_costs) normalized cost advantages.
      returns_c_batch: (batch, num_costs) cost returns.
      adv_c_mean: (num_costs,) mean of cost advantages (before normalization).
      adv_c_std: (num_costs,) std of cost advantages (before normalization).
      cost_limits: (num_costs,) optional d_i threshold per cost.

    Returns:
      (L_viol_scalar, L_viol_per_cost) for logging.
    """
    # Eq. 7: L_{C_i}^CLIP — clipped cost surrogate
    cost_loss1 = adv_c_batch * ratio.unsqueeze(-1)
    cost_loss2 = adv_c_batch * clipped_ratio.unsqueeze(-1)
    L_clip_c = torch.max(cost_loss1, cost_loss2).mean(dim=0)  # (num_costs,)

    # Eq. 7: (1-γ)(J_{C_i}(π_k) - d_i) term, advantage-normalized
    if cost_limits is not None:
      batch_cost_ret = (1.0 - self.c_gamma) * (returns_c_batch.mean(dim=0) - cost_limits)
    else:
      batch_cost_ret = (1.0 - self.c_gamma) * returns_c_batch.mean(dim=0)
    batch_cost_ret = (batch_cost_ret + adv_c_mean) / (adv_c_std + 1e-8)

    # Eq. 5: κ · Σ max{0, L_{C_i}^CLIP}
    L_viol_per_cost = L_clip_c + batch_cost_ret  # (num_costs,)
    L_viol = (self.c_scale * torch.clamp(L_viol_per_cost, min=0.0)).sum()

    return L_viol, L_viol_per_cost.detach()

  def _compute_cost_value_loss(
    self,
    new_values_c: torch.Tensor,
    old_values_c: torch.Tensor,
    returns_c: torch.Tensor,
    clip_param: float,
  ) -> torch.Tensor:
    """Clipped MSE cost value loss (same structure as reward value loss)."""
    v_loss_unclipped = (new_values_c - returns_c) ** 2
    v_clipped = old_values_c + (new_values_c - old_values_c).clamp(-clip_param, clip_param)
    v_loss_clipped = (v_clipped - returns_c) ** 2
    return 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean(dim=0).sum()

  def _step_adaptive_kappa(self):
    """Algorithm 2 line 5: κ ← min{ρκ, κ_max}."""
    if self.adaptive_kappa:
      self.c_scale = torch.clamp(self.c_scale * self.kappa_rho, max=self.kappa_max)

  def _normalize_cost_advantages(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-cost advantage normalization (P3O Section 5 trick).

    Returns:
      (adv_c_mean, adv_c_std) before normalization, for use in L_viol.
    """
    adv_c = self.storage.advantages_c
    # Flatten: (T, N, num_costs) -> (T*N, num_costs)
    flat = adv_c.reshape(-1, self.num_costs)
    adv_c_mean = flat.mean(dim=0)  # (num_costs,)
    adv_c_std = flat.std(dim=0)  # (num_costs,)
    # Normalize in-place
    self.storage.advantages_c = (adv_c - adv_c_mean) / (adv_c_std + 1e-8)
    return adv_c_mean, adv_c_std
