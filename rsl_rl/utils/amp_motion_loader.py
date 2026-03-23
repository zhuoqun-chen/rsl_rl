"""Generic AMP motion loader for expert demonstration data.

Ported from humanoid_skateboarding G1_AMPLoader, generalized with configurable obs extraction.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


class AMPMotionLoader:
  """Loads expert motion trajectories and provides multi-frame mini-batches for AMP training.

  This is a generic loader that works with flat numpy arrays. For structured motion data
  (e.g., body_pos_w + body_quat_w from .npz files), use a task-specific adapter that
  implements the same `feed_forward_generator` interface.

  Args:
    device: Torch device.
    motion_files: Comma-separated paths or single path to .npy/.npz motion files.
    amp_obs_dim: Per-frame AMP observation dimension.
    amp_num_frames: Number of consecutive frames per sample.
    time_between_frames: Time interval between consecutive frames (seconds).
    obs_extract_fn: Callable to extract per-frame AMP obs from full frame data.
      Signature: (full_frame: np.ndarray of shape (obs_full_dim,)) -> np.ndarray of shape (amp_obs_dim,).
      If None, uses full frame as-is (identity).
    num_preload_transitions: Number of transitions to preload at init.
  """

  def __init__(
    self,
    device: str | torch.device,
    motion_files: str | list[str],
    amp_obs_dim: int,
    amp_num_frames: int,
    time_between_frames: float = 1 / 50.0,
    obs_extract_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    num_preload_transitions: int = 200000,
  ):
    self.device = device
    self.amp_obs_dim = amp_obs_dim
    self.amp_num_frames = amp_num_frames
    self.time_between_frames = time_between_frames
    self.obs_extract_fn = obs_extract_fn or (lambda x: x)

    if isinstance(motion_files, str):
      motion_files = [f.strip() for f in motion_files.split(",") if f.strip()]
    self.motion_files = motion_files

    # Load all trajectories
    self.trajectories: list[np.ndarray] = []
    self.trajectory_lens: list[int] = []
    self.trajectory_weights: list[float] = []

    for path in self.motion_files:
      data = np.load(path, allow_pickle=True)
      if isinstance(data, np.lib.npyio.NpzFile):
        # Expect a 'frames' key or use the first array
        if "frames" in data:
          frames = data["frames"]
        else:
          keys = list(data.keys())
          frames = data[keys[0]]
      else:
        frames = data

      # Apply obs extraction to each frame
      extracted = np.array([self.obs_extract_fn(frame) for frame in frames])
      self.trajectories.append(extracted)
      self.trajectory_lens.append(len(extracted))

    # Compute sampling weights proportional to trajectory length
    total_len = sum(self.trajectory_lens)
    self.trajectory_weights = [l / total_len for l in self.trajectory_lens]

    # Preload multi-frame transitions
    self._preloaded: torch.Tensor | None = None
    if num_preload_transitions > 0:
      self._preload(num_preload_transitions)

  def _preload(self, num_transitions: int) -> None:
    """Preload multi-frame transitions for efficient sampling."""
    all_transitions = []
    for _ in range(num_transitions):
      # Sample a trajectory
      traj_idx = np.random.choice(len(self.trajectories), p=self.trajectory_weights)
      traj = self.trajectories[traj_idx]
      traj_len = self.trajectory_lens[traj_idx]

      # Sample a starting frame (ensure we have enough frames for the window)
      max_start = max(0, traj_len - self.amp_num_frames)
      start_idx = np.random.randint(0, max_start + 1)

      # Extract multi-frame window
      frames = traj[start_idx : start_idx + self.amp_num_frames]
      # Pad if not enough frames
      if len(frames) < self.amp_num_frames:
        pad = np.zeros((self.amp_num_frames - len(frames), self.amp_obs_dim), dtype=np.float32)
        frames = np.concatenate([pad, frames], axis=0)

      all_transitions.append(frames)

    self._preloaded = torch.tensor(np.array(all_transitions), dtype=torch.float32, device=self.device)

  def feed_forward_generator(self, num_mini_batches: int, mini_batch_size: int):
    """Yield mini-batches of shape (mini_batch_size, amp_num_frames, amp_obs_dim).

    If preloaded, samples from preloaded buffer. Otherwise generates on-the-fly.
    """
    for _ in range(num_mini_batches):
      if self._preloaded is not None:
        idxs = np.random.choice(len(self._preloaded), size=mini_batch_size)
        yield self._preloaded[idxs]
      else:
        batch = []
        for _ in range(mini_batch_size):
          traj_idx = np.random.choice(len(self.trajectories), p=self.trajectory_weights)
          traj = self.trajectories[traj_idx]
          traj_len = self.trajectory_lens[traj_idx]
          max_start = max(0, traj_len - self.amp_num_frames)
          start_idx = np.random.randint(0, max_start + 1)
          frames = traj[start_idx : start_idx + self.amp_num_frames]
          if len(frames) < self.amp_num_frames:
            pad = np.zeros((self.amp_num_frames - len(frames), self.amp_obs_dim), dtype=np.float32)
            frames = np.concatenate([pad, frames], axis=0)
          batch.append(frames)
        yield torch.tensor(np.array(batch), dtype=torch.float32, device=self.device)
