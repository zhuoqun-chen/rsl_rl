"""ConstrainedPPO: PPO + P3O cost constraints.

Composes CostConstraintMixin with PPO. Overrides act(), process_env_step(),
compute_returns(), init_storage(), and update() to add cost handling.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms.cost_constraint_mixin import CostConstraintMixin
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.storage import RolloutStorage


class ConstrainedPPO(CostConstraintMixin, PPO):
  """PPO with P3O cost penalty. No AMP."""

  def __init__(
    self,
    policy,
    num_costs: int = 0,
    c_gamma: list[float] | None = None,
    c_scale: list[float] | None = None,
    cost_value_loss_coef: float = 1.0,
    adaptive_kappa: bool = True,
    kappa_rho: float = 1.5,
    kappa_max: float = 100.0,
    normalize_cost: bool = True,
    cost_limits: list[float] | None = None,
    cost_term_names: list[str] | None = None,
    **ppo_kwargs,
  ):
    PPO.__init__(self, policy, **ppo_kwargs)
    if c_gamma is None:
      c_gamma = [0.99] * num_costs
    if c_scale is None:
      c_scale = [1.0] * num_costs
    self._init_cost_constraint(
      num_costs, c_gamma, c_scale, cost_value_loss_coef,
      adaptive_kappa, kappa_rho, kappa_max, normalize_cost,
      cost_term_names,
    )
    if cost_limits is not None:
      self.cost_limits = torch.tensor(cost_limits, dtype=torch.float32, device=self.device)
    else:
      self.cost_limits = torch.zeros(num_costs, device=self.device)

  def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
    self.storage = RolloutStorage(
      training_type, num_envs, num_transitions_per_env, obs, actions_shape,
      self.device, num_costs=self.num_costs,
    )

  def act(self, obs):
    actions = PPO.act(self, obs)
    if self.num_costs > 0:
      self.transition.values_c = self.policy.evaluate_costs(obs).detach()
    return actions

  def process_env_step(self, obs, rewards, dones, extras):
    if self.num_costs > 0 and "costs" in extras:
      self.transition.costs = extras["costs"].to(self.device)
    PPO.process_env_step(self, obs, rewards, dones, extras)

  def compute_returns(self, obs):
    PPO.compute_returns(self, obs)
    if self.num_costs > 0:
      last_values_c = self.policy.evaluate_costs(obs).detach()
      self.storage.compute_cost_returns(last_values_c, self.c_gamma, self.lam)

  def update(self):  # noqa: C901
    mean_value_loss = 0
    mean_surrogate_loss = 0
    mean_entropy = 0
    mean_cost_loss = 0
    mean_cost_value_loss = 0
    mean_l_viol_per_cost = torch.zeros(self.num_costs, device=self.device) if self.num_costs > 0 else None
    if self.rnd:
      mean_rnd_loss = 0
    else:
      mean_rnd_loss = None
    if self.symmetry:
      mean_symmetry_loss = 0
    else:
      mean_symmetry_loss = None

    # Normalize cost advantages and get pre-normalization stats for L_viol
    if self.num_costs > 0:
      adv_c_mean, adv_c_std = self._normalize_cost_advantages()

    # Mini-batch generator
    if self.policy.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    for sample in generator:
      # Unpack standard PPO tensors (first 10 elements)
      (
        obs_batch,
        actions_batch,
        target_values_batch,
        advantages_batch,
        returns_batch,
        old_actions_log_prob_batch,
        old_mu_batch,
        old_sigma_batch,
        hid_states_batch,
        masks_batch,
      ) = sample[:10]

      # Unpack cost tensors if present
      if self.num_costs > 0 and len(sample) > 10:
        costs_batch, values_c_batch, returns_c_batch, advantages_c_batch = sample[10:14]
      else:
        costs_batch = values_c_batch = returns_c_batch = advantages_c_batch = None

      num_aug = 1
      original_batch_size = obs_batch.batch_size[0]

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          advantages_batch = (advantages_batch - advantages_batch.mean()) / (
            advantages_batch.std() + 1e-8
          )

      # Symmetric augmentation
      if self.symmetry and self.symmetry["use_data_augmentation"]:
        data_augmentation_func = self.symmetry["data_augmentation_func"]
        obs_batch, actions_batch = data_augmentation_func(
          obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"],
        )
        num_aug = int(obs_batch.batch_size[0] / original_batch_size)
        old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
        target_values_batch = target_values_batch.repeat(num_aug, 1)
        advantages_batch = advantages_batch.repeat(num_aug, 1)
        returns_batch = returns_batch.repeat(num_aug, 1)

      # Recompute actions log prob and entropy
      self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
      actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
      value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
      mu_batch = self.policy.action_mean[:original_batch_size]
      sigma_batch = self.policy.action_std[:original_batch_size]
      entropy_batch = self.policy.entropy[:original_batch_size]

      # KL-based LR adaptation
      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = torch.sum(
            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
            / (2.0 * torch.square(sigma_batch))
            - 0.5,
            axis=-1,
          )
          kl_mean = torch.mean(kl)
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      # Surrogate loss (Eq. 6: L_R^CLIP)
      ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
      surrogate = -torch.squeeze(advantages_batch) * ratio
      clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
      surrogate_clipped = -torch.squeeze(advantages_batch) * clipped_ratio
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      # Value function loss
      if self.use_clipped_value_loss:
        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (returns_batch - value_batch).pow(2).mean()

      loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

      # P3O cost constraint loss (Eq. 5: L^P3O)
      cost_loss = torch.tensor(0.0, device=self.device)
      cost_value_loss = torch.tensor(0.0, device=self.device)
      if self.num_costs > 0 and advantages_c_batch is not None:
        # L_viol (Eq. 5 + 7)
        cost_loss, l_viol_per_cost = self._compute_cost_loss(
          ratio, clipped_ratio, advantages_c_batch, returns_c_batch,
          adv_c_mean, adv_c_std, self.cost_limits,
        )
        loss += cost_loss

        # Cost value function loss
        new_values_c = self.policy.evaluate_costs(obs_batch)
        cost_value_loss = self._compute_cost_value_loss(
          new_values_c, values_c_batch, returns_c_batch, self.clip_param,
        )
        loss += self.cost_value_loss_coef * cost_value_loss

      # Symmetry loss
      if self.symmetry:
        if not self.symmetry["use_data_augmentation"]:
          data_augmentation_func = self.symmetry["data_augmentation_func"]
          obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
          num_aug = int(obs_batch.shape[0] / original_batch_size)
        mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
        action_mean_orig = mean_actions_batch[:original_batch_size]
        _, actions_mean_symm_batch = data_augmentation_func(
          obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
        )
        mse_loss_fn = torch.nn.MSELoss()
        symmetry_loss = mse_loss_fn(
          mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
        )
        if self.symmetry["use_mirror_loss"]:
          loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
        else:
          symmetry_loss = symmetry_loss.detach()

      # RND loss
      if self.rnd:
        with torch.no_grad():
          rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
          rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
        predicted_embedding = self.rnd.predictor(rnd_state_batch)
        target_embedding = self.rnd.target(rnd_state_batch).detach()
        rnd_loss = torch.nn.MSELoss()(predicted_embedding, target_embedding)

      # Backward pass
      self.optimizer.zero_grad()
      loss.backward()
      if self.rnd:
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
      self.optimizer.step()
      if self.rnd_optimizer:
        self.rnd_optimizer.step()

      # Accumulate losses
      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy_batch.mean().item()
      mean_cost_loss += cost_loss.item()
      mean_cost_value_loss += cost_value_loss.item()
      if mean_l_viol_per_cost is not None and self.num_costs > 0 and advantages_c_batch is not None:
        mean_l_viol_per_cost += l_viol_per_cost
      if mean_rnd_loss is not None:
        mean_rnd_loss += rnd_loss.item()
      if mean_symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()

    # Adaptive κ: step at end of each outer iteration (Algorithm 2)
    if self.num_costs > 0:
      self._step_adaptive_kappa()

    # Average losses
    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    mean_cost_loss /= num_updates
    mean_cost_value_loss /= num_updates
    if mean_l_viol_per_cost is not None:
      mean_l_viol_per_cost /= num_updates
    if mean_rnd_loss is not None:
      mean_rnd_loss /= num_updates
    if mean_symmetry_loss is not None:
      mean_symmetry_loss /= num_updates
    self.storage.clear()

    loss_dict = {
      "value_function": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
      "cost_L_viol": mean_cost_loss,
      "cost_value": mean_cost_value_loss,
    }
    if mean_l_viol_per_cost is not None:
      for i in range(self.num_costs):
        name = self.cost_term_names[i]
        loss_dict[f"cost/{name}/L_viol"] = mean_l_viol_per_cost[i].item()
    if self.adaptive_kappa:
      loss_dict["kappa_mean"] = self.c_scale.mean().item()
    if self.rnd:
      loss_dict["rnd"] = mean_rnd_loss
    if self.symmetry:
      loss_dict["symmetry"] = mean_symmetry_loss
    return loss_dict
