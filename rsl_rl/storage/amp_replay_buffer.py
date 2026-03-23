"""AMP replay buffer for storing multi-frame policy observations.

Ported from humanoid_skateboarding ReplayBufferMulti with standardized naming.
"""

from __future__ import annotations

import numpy as np
import torch


class AMPReplayBuffer:
  """Fixed-size circular buffer for multi-frame AMP observation states."""

  def __init__(self, amp_obs_dim: int, buffer_size: int, amp_num_frames: int, device: str | torch.device):
    self.states = torch.zeros(buffer_size, amp_num_frames, amp_obs_dim).to(device)
    self.amp_num_frames = amp_num_frames
    self.buffer_size = buffer_size
    self.device = device

    self.step = 0
    self.num_samples = 0

  def insert(self, states: torch.Tensor) -> None:
    """Add new multi-frame states to buffer.

    Args:
      states: (batch, amp_num_frames, amp_obs_dim)
    """
    num_states = states.shape[0]
    start_idx = self.step
    end_idx = self.step + num_states
    if end_idx > self.buffer_size:
      self.states[self.step : self.buffer_size] = states[: self.buffer_size - self.step]
      self.states[: end_idx - self.buffer_size] = states[self.buffer_size - self.step :]
    else:
      self.states[start_idx:end_idx] = states

    self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
    self.step = (self.step + num_states) % self.buffer_size

  def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
    """Yield random mini-batches of (batch, amp_num_frames, amp_obs_dim)."""
    for _ in range(num_mini_batch):
      sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
      yield self.states[sample_idxs].to(self.device)
