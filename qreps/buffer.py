import collections
import random

import dm_env
import numpy as np
import torch
from bsuite.baselines import base
from bsuite.baselines.utils.sequence import Trajectory


class ReplayBuffer(object):
    """A simple Python replay buffer."""

    def __init__(self, capacity, discount=1.0):
        self._prev = None
        self._action = None
        self._latest = None
        self.buffer = collections.deque(maxlen=capacity)
        self.discount = discount

    def push(self, env_output, action):
        self._prev = self._latest
        self._action = action
        self._latest = env_output

        if self._prev is not None:
            self.buffer.append(
                (
                    self._prev.observation,
                    self._action,
                    self._latest.reward,
                    self._latest.discount,
                    self._latest.observation,
                )
            )

        if self._latest.last:
            self._latest = None

    def sample(self, batch_size):
        obs_tm1, a_tm1, r_t, discount_t, obs_t = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            torch.tensor(obs_tm1),
            torch.tensor(a_tm1),
            torch.tensor(r_t),
            torch.tensor(discount_t) * self.discount,
            torch.tensor(obs_t),
        )

    def is_ready(self, batch_size):
        return batch_size <= len(self.buffer)

    def reset(self):
        self.buffer.clear()
        self._prev = None


class Buffer:
    """A simple buffer for accumulating trajectories."""

    _observations: []
    _actions: []
    _rewards: []
    _discounts: []

    _max_sequence_length: int
    _needs_reset: bool = True
    _t: int = 0

    def __init__(
        self, max_sequence_length: int,
    ):
        """Pre-allocates buffers of numpy arrays to hold the sequences."""
        self._observations = [None] * (max_sequence_length + 1)
        self._actions = [None] * max_sequence_length
        self._rewards = np.zeros(max_sequence_length, dtype=np.float32)
        self._discounts = np.zeros(max_sequence_length, dtype=np.float32)

        self._max_sequence_length = max_sequence_length

    def append(
        self,
        timestep: dm_env.TimeStep,
        action: base.Action,
        new_timestep: dm_env.TimeStep,
    ):
        """Appends an observation, action, reward, and discount to the buffer."""
        if self.full():
            raise ValueError("Cannot append; sequence buffer is full.")

        # Start a new sequence with an initial observation, if required.
        if self._needs_reset:
            self._t = 0
            self._observations[self._t] = timestep.observation
            self._needs_reset = False

        # Append (o, a, r, d) to the sequence buffer.
        self._observations[self._t + 1] = new_timestep.observation
        self._actions[self._t] = action
        self._rewards[self._t] = new_timestep.reward
        self._discounts[self._t] = new_timestep.discount
        self._t += 1

        # Don't accumulate sequences that cross episode boundaries.
        # It is up to the caller to drain the buffer in this case.
        if new_timestep.last():
            self._needs_reset = True

    def drain(self) -> Trajectory:
        """Empties the buffer and returns the (possibly partial) trajectory."""
        if self.empty():
            raise ValueError("Cannot drain; sequence buffer is empty.")
        trajectory = Trajectory(
            self._observations[: self._t + 1],
            self._actions[: self._t],
            self._rewards[: self._t],
            self._discounts[: self._t],
        )
        self._t = 0  # Mark sequences as consumed.
        self._needs_reset = True
        return trajectory

    def empty(self) -> bool:
        """Returns whether or not the trajectory buffer is empty."""
        return self._t == 0

    def full(self) -> bool:
        """Returns whether or not the trajectory buffer is full."""
        return self._t == self._max_sequence_length
