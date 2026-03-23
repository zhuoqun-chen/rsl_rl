"""AMP on-policy runner extending OnPolicyRunner with AMP reward and discriminator.

Design difference from reference: extends OnPolicyRunner instead of being standalone.
Overrides __init__, learn, save, load, train_mode, eval_mode, _construct_algorithm.
Fixes reference bug: persists discriminator and normalizer state in save/load.
"""

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque
from typing import Callable

from rsl_rl.algorithms.amp_ppo import AMP_PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.modules.amp_discriminator import AMPDiscriminator
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import resolve_obs_groups, store_code_state
from rsl_rl.utils.amp_normalizer import AMPNormalizer


class AMPOnPolicyRunner(OnPolicyRunner):
  """On-policy runner with AMP (Adversarial Motion Priors) style reward.

  Pre-constructs AMP components (discriminator, normalizer, motion loader) before
  calling super().__init__ which triggers _construct_algorithm.
  """

  def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
    # Extract AMP config before super().__init__ consumes train_cfg
    self.amp_obs_dim: int = train_cfg["amp_obs_dim"]
    self.amp_num_frames: int = train_cfg.get("amp_num_frames", 5)

    # Build AMP components that _construct_algorithm will need
    self.amp_normalizer = AMPNormalizer(self.amp_obs_dim)
    self.discriminator = AMPDiscriminator(
      self.amp_obs_dim,
      train_cfg.get("amp_reward_coef", 2.0),
      train_cfg.get("amp_discr_hidden_dims", (256, 256)),
      device,
      self.amp_num_frames,
      train_cfg.get("amp_task_reward_lerp", 0.5),
      train_cfg.get("use_lerp", True),
    ).to(device)

    # amp_data must be set by subclass or passed in train_cfg["amp_data"]
    # This is a hook for task-specific motion loaders (e.g., MotionLoaderAdapter)
    self.amp_data = train_cfg.pop("amp_data", None)

    # Gating function (None = all envs get AMP reward)
    amp_gate_fn_cfg = train_cfg.get("amp_gate_fn", None)
    self.amp_gate_fn: Callable | None = None
    if amp_gate_fn_cfg is not None and isinstance(amp_gate_fn_cfg, str):
      from rsl_rl.utils import string_to_callable
      self.amp_gate_fn = string_to_callable(amp_gate_fn_cfg)

    # Now call parent which will call _construct_algorithm
    super().__init__(env, train_cfg, log_dir, device)

  def _construct_algorithm(self, obs) -> AMP_PPO:
    """Construct AMP_PPO algorithm with discriminator and motion data."""
    # resolve RND config
    self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
    # resolve symmetry config
    self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

    # resolve deprecated normalization config
    if self.cfg.get("empirical_normalization") is not None:
      if self.policy_cfg.get("actor_obs_normalization") is None:
        self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
      if self.policy_cfg.get("critic_obs_normalization") is None:
        self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

    # initialize the actor-critic
    actor_critic_class = eval(self.policy_cfg.pop("class_name"))
    actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
      obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
    ).to(self.device)

    # Extract AMP-specific params from alg_cfg before passing to AMP_PPO
    alg_cfg = dict(self.alg_cfg)
    alg_class_name = alg_cfg.pop("class_name", "AMP_PPO")
    amp_replay_buffer_size = alg_cfg.pop("amp_replay_buffer_size", 100000)
    amp_separate_optimizer = alg_cfg.pop("amp_separate_optimizer", False)
    amp_disc_learning_rate = alg_cfg.pop("amp_disc_learning_rate", 1e-3)

    # Remove AMP runner-level keys that PPO doesn't understand
    for key in list(alg_cfg.keys()):
      if key.startswith("amp_") or key in ("use_lerp",):
        alg_cfg.pop(key)

    # Initialize AMP_PPO
    alg: AMP_PPO = AMP_PPO(
      actor_critic,
      discriminator=self.discriminator,
      amp_data=self.amp_data,
      amp_normalizer=self.amp_normalizer,
      amp_num_frames=self.amp_num_frames,
      amp_replay_buffer_size=amp_replay_buffer_size,
      amp_separate_optimizer=amp_separate_optimizer,
      amp_disc_learning_rate=amp_disc_learning_rate,
      device=self.device,
      multi_gpu_cfg=self.multi_gpu_cfg,
      **alg_cfg,
    )

    # Initialize storage
    alg.init_storage(
      "rl",
      self.env.num_envs,
      self.num_steps_per_env,
      obs,
      [self.env.num_actions],
    )

    return alg

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
    self._prepare_logging_writer()

    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self.device)
    self.train_mode()

    # Initialize AMP observation sliding window
    amp_obs = self.env.get_amp_observations().to(self.device)
    self.amp_obs_frames = torch.zeros(
      self.env.num_envs, self.amp_num_frames, self.amp_obs_dim, device=self.device
    )
    self.amp_obs_frames = torch.cat(
      (self.amp_obs_frames[:, 1:], amp_obs.unsqueeze(1)), dim=1
    )

    # Book keeping
    ep_infos = []
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    step_discrewbuffer = deque(maxlen=100)

    cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
    cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
    cur_single_step_disc_rew = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

    if self.alg.rnd:
      erewbuffer = deque(maxlen=100)
      irewbuffer = deque(maxlen=100)
      cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
      cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()

    start_iter = self.current_learning_iteration
    tot_iter = start_iter + num_learning_iterations
    for it in range(start_iter, tot_iter):
      start = time.time()
      # Rollout
      with torch.inference_mode():
        for _ in range(self.num_steps_per_env):
          actions = self.alg.act(obs, amp_obs)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)

          # Get next AMP observations
          next_amp_obs = self.env.get_amp_observations().to(self.device)

          # Derive reset env IDs from dones (our wrapper doesn't expose reset_env_ids)
          reset_env_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)

          # Slide AMP observation window
          self.amp_obs_frames = torch.cat(
            (self.amp_obs_frames[:, 1:], next_amp_obs.unsqueeze(1)), dim=1
          )

          # Compute AMP style reward
          amp_reward = torch.zeros(self.env.num_envs, device=self.device)
          if self.amp_gate_fn is not None:
            mask = self.amp_gate_fn(obs)
            if mask.any():
              rewards[mask], _, disc_rew = self.discriminator.predict_amp_reward(
                self.amp_obs_frames[mask], rewards[mask], normalizer=self.amp_normalizer
              )
              amp_reward[mask] += disc_rew
          else:
            rewards, _, disc_rew = self.discriminator.predict_amp_reward(
              self.amp_obs_frames, rewards, normalizer=self.amp_normalizer
            )
            amp_reward += disc_rew

          # Process step
          self.alg.process_env_step(obs, rewards, dones, extras, next_amp_obs, self.amp_obs_frames)

          # Zero out AMP frames for reset envs
          if reset_env_ids.numel() > 0:
            self.amp_obs_frames[reset_env_ids] = 0

          amp_obs = next_amp_obs

          # Logging
          intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
          if self.log_dir is not None:
            if "episode" in extras:
              ep_infos.append(extras["episode"])
            elif "log" in extras:
              ep_infos.append(extras["log"])
            if self.alg.rnd:
              cur_ereward_sum += rewards
              cur_ireward_sum += intrinsic_rewards
              cur_reward_sum += rewards + intrinsic_rewards
            else:
              cur_reward_sum += rewards
            cur_episode_length += 1
            cur_single_step_disc_rew += amp_reward
            new_ids = (dones > 0).nonzero(as_tuple=False)
            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0
            if new_ids.numel() > 0 and hasattr(self.env, "max_episode_length_s"):
              to_extend = (cur_single_step_disc_rew[new_ids] / self.env.max_episode_length_s)[:, 0].cpu().numpy()
              step_discrewbuffer.extend(to_extend.tolist())
            elif new_ids.numel() > 0:
              step_discrewbuffer.extend(cur_single_step_disc_rew[new_ids][:, 0].cpu().numpy().tolist())
            cur_single_step_disc_rew[new_ids] = 0
            if self.alg.rnd:
              erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
              irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
              cur_ereward_sum[new_ids] = 0
              cur_ireward_sum[new_ids] = 0

        stop = time.time()
        collection_time = stop - start
        start = stop

        self.alg.compute_returns(obs)

      loss_dict = self.alg.update()

      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it
      if self.log_dir is not None and not self.disable_logs:
        self.log(locals())
        if it % self.save_interval == 0:
          self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

      ep_infos.clear()
      if it == start_iter and not self.disable_logs:
        git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
        if self.logger_type in ["wandb", "neptune"] and git_file_paths:
          for path in git_file_paths:
            self.writer.save_file(path)

    if self.log_dir is not None and not self.disable_logs:
      self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

  def log(self, locs: dict, width: int = 80, pad: int = 35):
    """Extended logging with AMP disc reward."""
    super().log(locs, width, pad)
    # Log AMP-specific metrics
    if self.log_dir is not None and self.writer is not None:
      if len(locs.get("step_discrewbuffer", [])) > 0:
        self.writer.add_scalar(
          "Train/mean_step_disc_reward",
          statistics.mean(locs["step_discrewbuffer"]),
          locs["it"],
        )

  def save(self, path: str, infos=None):
    """Save model with discriminator and normalizer state (fixes reference bug)."""
    saved_dict = {
      "model_state_dict": self.alg.policy.state_dict(),
      "optimizer_state_dict": self.alg.optimizer.state_dict(),
      "discriminator_state_dict": self.alg.discriminator.state_dict(),
      "amp_normalizer": self.alg.amp_normalizer,
      "iter": self.current_learning_iteration,
      "infos": infos,
    }
    if self.alg.amp_separate_optimizer and self.alg.disc_optimizer is not None:
      saved_dict["disc_optimizer_state_dict"] = self.alg.disc_optimizer.state_dict()
    if hasattr(self.alg, "rnd") and self.alg.rnd:
      saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
      saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
    torch.save(saved_dict, path)

    if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
      self.writer.save_model(path, self.current_learning_iteration)

  def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
    """Load model with discriminator and normalizer state."""
    loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
    resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])

    # Load discriminator state
    if "discriminator_state_dict" in loaded_dict:
      self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
    # Load AMP normalizer
    if "amp_normalizer" in loaded_dict:
      self.alg.amp_normalizer = loaded_dict["amp_normalizer"]
      self.amp_normalizer = loaded_dict["amp_normalizer"]

    if hasattr(self.alg, "rnd") and self.alg.rnd:
      self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])

    if load_optimizer and resumed_training:
      self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
      if (
        self.alg.amp_separate_optimizer
        and self.alg.disc_optimizer is not None
        and "disc_optimizer_state_dict" in loaded_dict
      ):
        self.alg.disc_optimizer.load_state_dict(loaded_dict["disc_optimizer_state_dict"])
      if hasattr(self.alg, "rnd") and self.alg.rnd:
        self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])

    if resumed_training:
      self.current_learning_iteration = loaded_dict["iter"]
    return loaded_dict["infos"]

  def train_mode(self):
    super().train_mode()
    self.alg.discriminator.train()

  def eval_mode(self):
    super().eval_mode()
    self.alg.discriminator.eval()
