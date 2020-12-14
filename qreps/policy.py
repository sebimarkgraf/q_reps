from abc import ABC, abstractmethod

import torch
from bsuite.baselines.utils import sequence


class Policy(ABC):
    @abstractmethod
    def sample(self, features):
        """Sample the policy"""
        pass

    @abstractmethod
    def fit(self, trajectory: sequence.Trajectory, weights: torch.Tensor):
        """Fit the policy to the provided samples"""
        pass


class DiscreteStochasticPolicy(Policy):
    def __init__(self, n_states, n_actions):
        self._policy = torch.rand((n_states, n_actions))

    def sample(self, observation):
        """Expect feature to just be the state"""
        state = int(observation.item())
        m = torch.distributions.Categorical(self._policy[state])
        return m.sample()

    def fit(self, trajectory: sequence.Trajectory, weights: torch.Tensor):
        states = trajectory.observations
        actions = trajectory.actions

        for s, a, w in zip(states, actions, weights):
            self._policy[s, a] = self._policy[s, a] * w

        self._policy /= torch.sum(self._policy, 1, keepdim=True)
