from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
import torch.nn as nn


class Policy(ABC):
    """Policy base class providing necessary interface for all derived policies.
    A policy provides two main mechanisms:
    * Sampling an action giving one observation necessary for running the policy
    * Updating the policy given a trajectory of transitions and weights for the transitions
    """

    def __init__(self, feature_fn=nn.Identity()):
        super(Policy, self).__init__()
        self._stochastic = True
        self.feature_fn = feature_fn

    @abstractmethod
    def sample(self, observation: torch.Tensor) -> Union[int, np.array]:
        """Sample the policy to obtain an action to perform"""
        pass

    @abstractmethod
    def log_likelihood(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        pass

    def set_eval_mode(self, enabled: bool):
        self._stochastic = not enabled

    def set_stochastic(self, enabled: bool):
        """Sets the policy to eval mode to enable exploitation without doing any more exploration"""
        self._stochastic = enabled

    def forward(self, x):
        return self.feature_fn(x)


class TorchStochasticPolicy(Policy, nn.Module):
    def __init__(self, n_states: int, n_actions: int, *args, **kwargs):
        super(TorchStochasticPolicy, self).__init__(*args, **kwargs)

        # Initialize with same prob for all actions in each state
        _policy = torch.ones((n_states, n_actions))
        _policy /= torch.sum(_policy, 1, keepdim=True)
        self._policy = nn.Parameter(_policy)

    def forward(self, x):
        return super(TorchStochasticPolicy, self).forward(x)

    def _dist(self, observation):
        return torch.distributions.Categorical(
            logits=self._policy[self.forward(observation)]
        )

    def log_likelihood(self, feat, taken_actions) -> torch.FloatTensor:
        return self._dist(feat).log_prob(taken_actions)

    @torch.no_grad()
    def sample(self, observation):
        if self._stochastic:
            return self._dist(observation).sample().item()
        else:
            return torch.argmax(self._policy[self.forward(observation)]).item()


class CategoricalMLP(Policy, torch.nn.Module):
    def __init__(self, obs_shape, act_shape, *args, **kwargs):
        super(CategoricalMLP, self).__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Linear(obs_shape, 128),
            nn.ReLU(),
            nn.Linear(128, act_shape),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.model(super(CategoricalMLP, self).forward(x))

    def _dist(self, observation) -> torch.distributions.Distribution:
        return torch.distributions.categorical.Categorical(self.forward(observation))

    @torch.no_grad()
    def sample(self, observation):
        if self._stochastic:
            return self._dist(observation).sample().item()
        else:
            return torch.argmax(self.forward(observation)).item()

    def log_likelihood(self, features, actions):
        return self._dist(features).log_prob(actions)


class GaussianMLP(Policy, torch.nn.Module):
    """Gaussian Multi Layer Perceptron as a Policy.

    Estimates mean of a gaussian distribution for every action and the corresponding deviation
    depending on the given observations.

    When set to eval mode returns the mean as action for every observation.
    """

    def __init__(
        self, obs_shape, act_shape, action_min, action_max, sigma=1.0, *args, **kwargs
    ):
        super(GaussianMLP, self).__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Linear(obs_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_shape),
        )
        self.log_sigma = nn.Parameter(torch.tensor(sigma))
        self.action_max = action_max
        self.action_min = action_min

    def forward(self, x):
        return self.model(super(GaussianMLP, self).forward(x))

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    def _dist(self, observation) -> torch.distributions.Distribution:
        return torch.distributions.normal.Normal(self.forward(observation), self.sigma)

    @torch.no_grad()
    def sample(self, observation):
        if self._stochastic:
            return torch.clamp(
                self._dist(observation).sample(), self.action_min, self.action_max
            )
        else:
            return torch.clamp(
                self.forward(observation), self.action_min, self.action_max
            )

    def log_likelihood(self, feat, taken_actions) -> torch.FloatTensor:
        return self._dist(feat).log_prob(taken_actions)
