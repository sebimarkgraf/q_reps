import collections
import random

import dm_env
import torch


class ReplayBuffer(object):
    """A simple Python replay buffer."""

    def __init__(self, capacity, discount=1.0):
        self._prev = None
        self._action = None
        self._latest = None
        self.buffer = collections.deque(maxlen=capacity)
        self.discount = discount

    def push(self, timestep: dm_env.TimeStep, action, new_timestep: dm_env.TimeStep):

        self.buffer.append(
            (
                new_timestep.observation,
                action,
                timestep.reward,
                timestep.discount,
                timestep.observation,
            )
        )

    def sample(self, batch_size):
        obs_tm1, a_tm1, r_t, discount_t, obs_t = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            torch.tensor(obs_tm1).float(),
            torch.tensor(a_tm1).float(),
            torch.tensor(r_t).float(),
            torch.tensor(discount_t).float() * self.discount,
            torch.tensor(obs_t).float(),
        )

    def get_all(self):
        obs_tm1, a_tm1, r_t, discount_t, obs_t = zip(*self.buffer)
        return (
            torch.tensor(obs_tm1).float(),
            torch.tensor(a_tm1).float(),
            torch.tensor(r_t).float(),
            torch.tensor(discount_t).float() * self.discount,
            torch.tensor(obs_t).float(),
        )

    def is_ready(self, batch_size):
        return batch_size <= len(self.buffer)

    def reset(self):
        self.buffer.clear()

    def full(self):
        return self.buffer.maxlen == len(self.buffer)
