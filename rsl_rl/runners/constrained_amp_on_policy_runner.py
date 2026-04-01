"""Constrained AMP on-policy runner: AMPOnPolicyRunner + cost data flow.

Extends AMPOnPolicyRunner with cost extraction, cost return computation,
and Episode_Cost/* logging. For AMPConstrainedP3O tasks.
"""

from __future__ import annotations

import warnings

from rsl_rl.algorithms.constrained_amp_ppo import ConstrainedAMP_PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.modules.actor_critic_cost import ActorCriticWithCostHeads
from rsl_rl.runners.amp_on_policy_runner import AMPOnPolicyRunner
from rsl_rl.storage import RolloutStorage


class ConstrainedAMPOnPolicyRunner(AMPOnPolicyRunner):
  """AMP on-policy runner with P3O cost constraint support."""

  def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
    self._num_costs = train_cfg.get("num_costs", 0)
    super().__init__(env, train_cfg, log_dir, device)

  def _construct_algorithm(self, obs) -> ConstrainedAMP_PPO:
    """Construct ConstrainedAMP_PPO with discriminator + cost value heads."""
    self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
    self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

    if self.cfg.get("empirical_normalization") is not None:
      if self.policy_cfg.get("actor_obs_normalization") is None:
        self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
      if self.policy_cfg.get("critic_obs_normalization") is None:
        self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

    # Build policy with cost heads
    policy_class_name = self.policy_cfg.pop("class_name")
    policy_class = eval(policy_class_name)
    if self._num_costs > 0 and policy_class is ActorCritic:
      policy_class = ActorCriticWithCostHeads

    policy_kwargs = dict(self.policy_cfg)
    alg_cfg = dict(self.alg_cfg)

    if self._num_costs > 0 and issubclass(policy_class, ActorCriticWithCostHeads):
      cost_hidden_dims = alg_cfg.pop("cost_hidden_dims", (256, 256))
      policy_kwargs["num_costs"] = self._num_costs
      policy_kwargs["cost_hidden_dims"] = cost_hidden_dims

    actor_critic = policy_class(
      obs, self.cfg["obs_groups"], self.env.num_actions, **policy_kwargs
    ).to(self.device)

    # Extract algorithm params
    alg_class_name = alg_cfg.pop("class_name", "ConstrainedAMP_PPO")

    # AMP-specific params
    amp_replay_buffer_size = alg_cfg.pop("amp_replay_buffer_size", 100000)
    amp_separate_optimizer = alg_cfg.pop("amp_separate_optimizer", False)
    amp_disc_learning_rate = alg_cfg.pop("amp_disc_learning_rate", 1e-3)

    # Cost-specific params
    num_costs = alg_cfg.pop("num_costs", self._num_costs)
    c_gamma = alg_cfg.pop("c_gamma", None)
    c_scale = alg_cfg.pop("c_scale", None)
    cost_value_loss_coef = alg_cfg.pop("cost_value_loss_coef", 1.0)
    adaptive_kappa = alg_cfg.pop("adaptive_kappa", True)
    kappa_rho = alg_cfg.pop("kappa_rho", 1.5)
    kappa_max = alg_cfg.pop("kappa_max", 100.0)
    normalize_cost = alg_cfg.pop("normalize_cost", True)
    cost_limits = alg_cfg.pop("cost_limits", None)
    alg_cfg.pop("c_gamma_overrides", None)
    alg_cfg.pop("c_scale_overrides", None)
    alg_cfg.pop("broadcast_cost_params", None)

    # Remove AMP runner-level keys
    for key in list(alg_cfg.keys()):
      if key.startswith("amp_") or key in ("use_lerp",):
        alg_cfg.pop(key)

    alg: ConstrainedAMP_PPO = ConstrainedAMP_PPO(
      actor_critic,
      discriminator=self.discriminator,
      amp_data=self.amp_data,
      amp_normalizer=self.amp_normalizer,
      amp_num_frames=self.amp_num_frames,
      amp_replay_buffer_size=amp_replay_buffer_size,
      amp_separate_optimizer=amp_separate_optimizer,
      amp_disc_learning_rate=amp_disc_learning_rate,
      num_costs=num_costs,
      c_gamma=c_gamma,
      c_scale=c_scale,
      cost_value_loss_coef=cost_value_loss_coef,
      adaptive_kappa=adaptive_kappa,
      kappa_rho=kappa_rho,
      kappa_max=kappa_max,
      normalize_cost=normalize_cost,
      cost_limits=cost_limits,
      device=self.device,
      multi_gpu_cfg=self.multi_gpu_cfg,
      **alg_cfg,
    )

    alg.init_storage(
      "rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions],
    )
    return alg
