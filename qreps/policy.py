from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
from bsuite.baselines.utils import sequence


class Policy(ABC):
    @abstractmethod
    def sample(self, observation: torch.Tensor) -> Union[int, np.array]:
        """Sample the policy"""
        pass

    @abstractmethod
    def fit(self, trajectory: sequence.Trajectory, weights: torch.Tensor) -> None:
        """Fit the policy to the provided samples"""
        pass


class DiscreteStochasticPolicy(Policy):
    def __init__(self, n_states: int, n_actions: int):
        self._policy = torch.ones((n_states, n_actions))
        self._policy /= torch.sum(self._policy, 1, keepdim=True)

    def sample(self, observation):
        """Expect observation to just be the state"""
        state = int(observation.item())
        m = torch.distributions.Categorical(self._policy[state])
        return int(m.sample())

    def fit(self, trajectory, weights):
        states = trajectory.observations
        actions = trajectory.actions

        for s, a, w in zip(states, actions, weights):
            self._policy[s, a] = self._policy[s, a] * w

        self._policy /= torch.sum(self._policy, 1, keepdim=True)


class GaussianMLP(Policy):
    def __init__(self, obs_shape, act_shape):
        self._mu = torch.zeros(act_shape)
        self._sigma = torch.ones(obs_shape)

    def sample(self, observation):
        """Expect feature to just be the state"""
        dist = torch.distributions.normal.Normal(self._mu, self._sigma)
        return dist.sample((1,))

    def fit(self, trajectory, weights):
        pass
