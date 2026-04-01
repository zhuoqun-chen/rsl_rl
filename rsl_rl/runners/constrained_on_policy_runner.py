"""Constrained on-policy runner extending OnPolicyRunner with cost data flow.

Reads num_costs from env, passes to algorithm/policy, extracts extras["costs"]
during rollout, computes cost returns, logs Episode_Cost/* metrics.
"""

from __future__ import annotations

import warnings

from rsl_rl.algorithms.constrained_ppo import ConstrainedPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.modules.actor_critic_cost import ActorCriticWithCostHeads
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class ConstrainedOnPolicyRunner(OnPolicyRunner):
  """On-policy runner with P3O cost constraint support."""

  def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
    # Extract num_costs from env before super().__init__
    self._num_costs = train_cfg.get("num_costs", 0)
    super().__init__(env, train_cfg, log_dir, device)

  def _construct_algorithm(self, obs) -> ConstrainedPPO:
    """Construct ConstrainedPPO with cost value heads."""
    self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
    self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

    if self.cfg.get("empirical_normalization") is not None:
      warnings.warn(
        "The `empirical_normalization` parameter is deprecated.",
        DeprecationWarning,
      )
      if self.policy_cfg.get("actor_obs_normalization") is None:
        self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
      if self.policy_cfg.get("critic_obs_normalization") is None:
        self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

    # Build policy with cost heads
    policy_class_name = self.policy_cfg.pop("class_name")
    policy_class = eval(policy_class_name)

    # If using cost heads, ensure we use ActorCriticWithCostHeads
    if self._num_costs > 0 and policy_class is ActorCritic:
      policy_class = ActorCriticWithCostHeads

    policy_kwargs = dict(self.policy_cfg)
    if self._num_costs > 0 and issubclass(policy_class, ActorCriticWithCostHeads):
      cost_hidden_dims = self.alg_cfg.pop("cost_hidden_dims", (256, 256))
      policy_kwargs["num_costs"] = self._num_costs
      policy_kwargs["cost_hidden_dims"] = cost_hidden_dims

    actor_critic = policy_class(
      obs, self.cfg["obs_groups"], self.env.num_actions, **policy_kwargs
    ).to(self.device)

    # Build algorithm
    alg_cfg = dict(self.alg_cfg)
    alg_class_name = alg_cfg.pop("class_name", "ConstrainedPPO")
    alg_class = eval(alg_class_name)

    # Extract cost-specific params
    num_costs = alg_cfg.pop("num_costs", self._num_costs)
    c_gamma = alg_cfg.pop("c_gamma", None)
    c_scale = alg_cfg.pop("c_scale", None)
    cost_value_loss_coef = alg_cfg.pop("cost_value_loss_coef", 1.0)
    adaptive_kappa = alg_cfg.pop("adaptive_kappa", True)
    kappa_rho = alg_cfg.pop("kappa_rho", 1.5)
    kappa_max = alg_cfg.pop("kappa_max", 100.0)
    normalize_cost = alg_cfg.pop("normalize_cost", True)
    cost_limits = alg_cfg.pop("cost_limits", None)
    cost_term_names = alg_cfg.pop("cost_term_names", None)
    # Remove keys that base PPO doesn't understand
    alg_cfg.pop("c_gamma_overrides", None)
    alg_cfg.pop("c_scale_overrides", None)
    alg_cfg.pop("broadcast_cost_params", None)
    # cost_hidden_dims already popped above

    alg: ConstrainedPPO = alg_class(
      actor_critic,
      num_costs=num_costs,
      c_gamma=c_gamma,
      c_scale=c_scale,
      cost_value_loss_coef=cost_value_loss_coef,
      adaptive_kappa=adaptive_kappa,
      kappa_rho=kappa_rho,
      kappa_max=kappa_max,
      normalize_cost=normalize_cost,
      cost_limits=cost_limits,
      cost_term_names=cost_term_names,
      device=self.device,
      multi_gpu_cfg=self.multi_gpu_cfg,
      **alg_cfg,
    )

    alg.init_storage(
      "rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions],
    )
    return alg

  def save(self, path: str, infos=None):
    """Save with cost value heads state."""
    super().save(path, infos)
    # Cost value heads are part of policy.state_dict(), so already saved by super()

  def load(self, path, load_optimizer=True, map_location=None):
    result = super().load(path, load_optimizer, map_location)
    return result
