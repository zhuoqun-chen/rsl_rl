"""AMP-PPO algorithm extending PPO with adversarial motion prior discriminator training.

Design difference from reference: extends PPO instead of being standalone, to inherit
upstream bugfixes and avoid code duplication. Overrides act(), process_env_step(), update().
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.amp_discriminator import AMPDiscriminator
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.amp_replay_buffer import AMPReplayBuffer
from rsl_rl.utils.amp_normalizer import AMPNormalizer


class AMP_PPO(PPO):
  """PPO with AMP discriminator for style reward.

  Extends PPO to add:
  - AMP replay buffer for policy-generated multi-frame states
  - Discriminator training alongside policy optimization
  - Configurable single or separate optimizer for policy and discriminator
  """

  def __init__(
    self,
    policy,
    discriminator: AMPDiscriminator,
    amp_data,
    amp_normalizer: AMPNormalizer,
    amp_num_frames: int = 1,
    amp_replay_buffer_size: int = 100000,
    amp_separate_optimizer: bool = False,
    amp_disc_learning_rate: float = 1e-3,
    **ppo_kwargs,
  ):
    super().__init__(policy, **ppo_kwargs)

    self.discriminator = discriminator
    self.discriminator.to(self.device)
    self.amp_storage = AMPReplayBuffer(
      discriminator.amp_obs_dim,
      amp_replay_buffer_size,
      amp_num_frames,
      self.device,
    )
    self.amp_data = amp_data
    self.amp_normalizer = amp_normalizer
    self.amp_transition = RolloutStorage.Transition()
    self.amp_separate_optimizer = amp_separate_optimizer

    if amp_separate_optimizer:
      # Independent optimizers: policy keeps its own (set by PPO.__init__),
      # discriminator gets a new one
      self.disc_optimizer = optim.Adam(
        [
          {"params": self.discriminator.trunk.parameters(), "weight_decay": 10e-4},
          {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 10e-2},
        ],
        lr=amp_disc_learning_rate,
      )
    else:
      # Single optimizer (reference approach): rebuild with policy + discriminator
      self.optimizer = optim.Adam(
        [
          {"params": self.policy.parameters(), "name": "policy"},
          {"params": self.discriminator.trunk.parameters(), "weight_decay": 10e-4, "name": "amp_trunk"},
          {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 10e-2, "name": "amp_head"},
        ],
        lr=self.learning_rate,
      )
      self.disc_optimizer = None

  def act(self, obs, amp_obs=None):
    """Sample actions. Optionally stores AMP observation for replay buffer."""
    if self.policy.is_recurrent:
      self.transition.hidden_states = self.policy.get_hidden_states()
    self.transition.actions = self.policy.act(obs).detach()
    self.transition.values = self.policy.evaluate(obs).detach()
    self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
    self.transition.action_mean = self.policy.action_mean.detach()
    self.transition.action_sigma = self.policy.action_std.detach()
    self.transition.observations = obs
    if amp_obs is not None:
      self.amp_transition.observations = amp_obs
    return self.transition.actions

  def process_env_step(self, obs, rewards, dones, extras, amp_obs=None, amp_obs_frames=None):
    """Process environment step. Inserts into AMP replay buffer, then standard PPO logic."""
    # Update normalizers
    self.policy.update_normalization(obs)
    if self.rnd:
      self.rnd.update_normalization(obs)

    self.transition.rewards = rewards.clone()
    self.transition.dones = dones

    # RND intrinsic rewards
    if self.rnd:
      self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
      self.transition.rewards += self.intrinsic_rewards

    # Bootstrapping on time outs
    if "time_outs" in extras:
      self.transition.rewards += self.gamma * torch.squeeze(
        self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
      )

    # Insert into AMP replay buffer
    if amp_obs_frames is not None:
      self.amp_storage.insert(amp_obs_frames)

    # Record transition and reset
    self.storage.add_transitions(self.transition)
    self.transition.clear()
    self.amp_transition.clear()
    self.policy.reset(dones)

  def update(self):  # noqa: C901
    mean_value_loss = 0
    mean_surrogate_loss = 0
    mean_entropy = 0
    mean_amp_loss = 0
    mean_grad_pen_loss = 0
    mean_policy_pred = 0
    mean_expert_pred = 0
    if self.rnd:
      mean_rnd_loss = 0
    else:
      mean_rnd_loss = None
    if self.symmetry:
      mean_symmetry_loss = 0
    else:
      mean_symmetry_loss = None

    # Mini-batch generators
    if self.policy.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    else:
      generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

    mini_batch_size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
    num_total_batches = self.num_learning_epochs * self.num_mini_batches

    amp_policy_generator = self.amp_storage.feed_forward_generator(num_total_batches, mini_batch_size)
    amp_expert_generator = self.amp_data.feed_forward_generator(num_total_batches, mini_batch_size)

    for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
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
      ) = sample

      num_aug = 1
      original_batch_size = obs_batch.batch_size[0]

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

      # Symmetric augmentation
      if self.symmetry and self.symmetry["use_data_augmentation"]:
        data_augmentation_func = self.symmetry["data_augmentation_func"]
        obs_batch, actions_batch = data_augmentation_func(
          obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"]
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

      # Surrogate loss
      ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
      surrogate = -torch.squeeze(advantages_batch) * ratio
      surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
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

      policy_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

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
        mse_loss = torch.nn.MSELoss()
        symmetry_loss = mse_loss(
          mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
        )
        if self.symmetry["use_mirror_loss"]:
          policy_loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
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

      # AMP discriminator loss
      expert_states = sample_amp_expert
      policy_states = sample_amp_policy

      with torch.no_grad():
        expert_states = self.amp_normalizer.normalize_torch(expert_states.to(self.device), self.device)
        policy_states = self.amp_normalizer.normalize_torch(policy_states, self.device)

      policy_d = self.discriminator(policy_states.flatten(1))
      expert_states = expert_states.to(self.device)
      expert_d = self.discriminator(expert_states.flatten(1))

      expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
      policy_d_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
      amp_loss = 0.5 * (expert_loss + policy_d_loss)
      grad_pen_loss = self.discriminator.compute_grad_pen(expert_states, lambda_=5)

      # Update normalizer with both policy and expert states
      self.amp_normalizer.update(policy_states.cpu().numpy().reshape(-1, self.discriminator.amp_obs_dim))
      self.amp_normalizer.update(expert_states.cpu().numpy().reshape(-1, self.discriminator.amp_obs_dim))

      # Backward pass and optimization
      if self.amp_separate_optimizer:
        # Two separate backward passes
        self.optimizer.zero_grad()
        policy_loss.backward()
        if self.rnd:
          self.rnd_optimizer.zero_grad()
          rnd_loss.backward()
        if self.is_multi_gpu:
          self.reduce_parameters()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.rnd_optimizer:
          self.rnd_optimizer.step()

        self.disc_optimizer.zero_grad()
        (amp_loss + grad_pen_loss).backward()
        self.disc_optimizer.step()
      else:
        # Single backward pass (reference approach)
        loss = policy_loss + amp_loss + grad_pen_loss
        self.optimizer.zero_grad()
        if self.rnd:
          self.rnd_optimizer.zero_grad()
          rnd_loss.backward()
        loss.backward()
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
      mean_amp_loss += amp_loss.item()
      mean_grad_pen_loss += grad_pen_loss.item()
      mean_policy_pred += policy_d_loss.mean().item()
      mean_expert_pred += expert_loss.mean().item()
      if mean_rnd_loss is not None:
        mean_rnd_loss += rnd_loss.item()
      if mean_symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()

    # Average losses
    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    mean_amp_loss /= num_updates
    mean_grad_pen_loss /= num_updates
    mean_policy_pred /= num_updates
    mean_expert_pred /= num_updates
    if mean_rnd_loss is not None:
      mean_rnd_loss /= num_updates
    if mean_symmetry_loss is not None:
      mean_symmetry_loss /= num_updates

    self.storage.clear()

    loss_dict = {
      "value_function": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
      "amp": mean_amp_loss,
      "amp_grad_pen": mean_grad_pen_loss,
      "amp_policy_pred": mean_policy_pred,
      "amp_expert_pred": mean_expert_pred,
    }
    if self.rnd:
      loss_dict["rnd"] = mean_rnd_loss
    if self.symmetry:
      loss_dict["symmetry"] = mean_symmetry_loss

    return loss_dict
