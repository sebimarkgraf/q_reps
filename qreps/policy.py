from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class Policy(ABC):
    """Policy base class providing necessary interface for all derived policies.
    A policy provides two main mechanisms:
    * Sampling an action giving one observation necessary for running the policy
    * Updating the policy given a trajectory of transitions and weights for the transitions
    """

    @abstractmethod
    def sample(self, observation: torch.Tensor) -> Union[int, np.array]:
        """Sample the policy to obtain an action to perform"""
        pass

    def fit(
        self, feat: torch.Tensor, actions: torch.Tensor, weights: torch.Tensor
    ) -> dict:
        """Fit the policy to the provided samples"""
        pass

    def set_eval_mode(self, enabled: bool):
        """Sets the policy to eval mode to enable exploitation without doing any more exploration"""
        pass


class TorchStochasticPolicy(Policy, nn.Module):
    def __init__(self, n_states: int, n_actions: int, minimizing_epochs=300):
        super().__init__()
        _policy = torch.ones((n_states, n_actions))
        _policy /= torch.sum(_policy, 1, keepdim=True)
        self._policy = nn.Parameter(_policy)
        self.stochastic = True
        self.lr = 1e-3
        self.opt = Adam(self.parameters(), lr=self.lr)
        self.minimizing_epochs = minimizing_epochs

    def _dist(self, observation):
        return torch.distributions.Categorical(logits=self._policy[observation])

    def log_likelihood(self, feat, taken_actions) -> torch.FloatTensor:
        return self._dist(feat).log_prob(taken_actions)

    def set_eval_mode(self, enabled):
        self.stochastic = not enabled

    @torch.no_grad()
    def sample(self, observation):
        if self.stochastic:
            return self._dist(observation).sample().item()
        else:
            return torch.argmax(self._policy[observation]).item()


class DiscreteStochasticPolicy(Policy):
    """Discrete Policy which assigns every action in every state a probability."""

    def __init__(self, n_states: int, n_actions: int):
        _policy = torch.ones((n_states, n_actions))
        _policy /= torch.sum(_policy, 1, keepdim=True)
        self._policy = _policy
        self._eval = False
        self.lr = 1.0

    def sample(self, observation):
        if self._eval:
            return int(torch.argmax(self._policy[observation]).item())

        m = self._dist(observation)
        return m.sample().item()

    def forward(self, observation):
        return self._policy[observation]

    def _dist(self, observation):
        return torch.distributions.Categorical(self._policy[observation])

    def fit(self, feats, actions, weights):
        log_like_before = self._dist(feats).log_prob(actions)

        for s, a, w in zip(feats.long().numpy(), actions.long().numpy(), weights):
            self._policy[s, a] = self._policy[s, a] * w

        self._policy = F.softmax(self._policy, dim=1)
        log_like_after = self._dist(feats).log_prob(actions)
        # FIXME: Return real loss
        return {
            "policy_loss": 0,
            "kl_loss": F.kl_div(
                log_like_before, log_like_after, log_target=True, reduction="batchmean"
            ),
        }

    def set_eval_mode(self, enabled: bool):
        self._eval = enabled


class CategoricalMLP(Policy, torch.nn.Module):
    def __init__(self, obs_shape, act_shape, minimizing_epochs=100, lr=1e-2):
        super(CategoricalMLP, self).__init__()
        self.minimizing_epochs = minimizing_epochs
        self.stochastic = True
        self.model = nn.Sequential(
            nn.Linear(obs_shape, 128),
            nn.ReLU(),
            nn.Linear(128, act_shape),
            nn.Softmax(dim=-1),
        )

        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.model(x)

    def _dist(self, observation) -> torch.distributions.Distribution:
        return torch.distributions.categorical.Categorical(self.forward(observation))

    @torch.no_grad()
    def sample(self, observation):
        if self.stochastic:
            return self._dist(observation).sample().item()
        else:
            return torch.argmax(self.forward(observation)).item()

    def fit(self, feat, actions, weights):
        loglike_before = self._dist(feat).log_prob(actions)

        for epoch in range(self.minimizing_epochs):
            self.opt.zero_grad()
            loss = self.log_likelihood(actions, feat, weights)
            loss.backward()
            self.opt.step()

        loglike_after = self._dist(feat).log_prob(actions)

        return {
            "policy_loss": loss,
            "kl_loss": F.kl_div(
                loglike_before, loglike_after, log_target=True, reduction="batchmean"
            ),
        }

    def log_likelihood(self, taken_actions, feat, weights) -> torch.FloatTensor:
        log_lik = self._dist(feat).log_prob(taken_actions)
        return -torch.mean(weights * log_lik)

    def set_eval_mode(self, enabled):
        self.stochastic = not enabled


class GaussianMLP(Policy, torch.nn.Module):
    """Gaussian Multi Layer Perceptron as a Policy.

    Estimates mean of a gaussian distribution for every action and the corresponding deviation
    depending on the given observations.

    When set to eval mode returns the mean as action for every observation.
    """

    def __init__(
        self,
        obs_shape,
        act_shape,
        action_min,
        action_max,
        sigma=1.0,
        minimizing_epochs=300,
        lr=1e-2,
    ):
        super(GaussianMLP, self).__init__()
        self.minimizing_epochs = minimizing_epochs
        self.model = nn.Sequential(
            nn.Linear(obs_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_shape),
        )
        self._sigma = nn.Parameter(torch.tensor(sigma))

        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.stochastic = True
        self.action_max = action_max
        self.action_min = action_min

    def forward(self, x):
        return self.model(x)

    def _dist(self, observation) -> torch.distributions.Distribution:
        return torch.distributions.normal.Normal(self.forward(observation), self._sigma)

    @torch.no_grad()
    def sample(self, observation):
        if self.stochastic:
            return torch.clamp(
                self._dist(observation).sample(), self.action_min, self.action_max
            )
        else:
            return torch.clamp(
                self.forward(observation), self.action_min, self.action_max
            )

    def fit(self, feat, actions, weights):
        loglike_before = self.log_likelihood(feat, actions)
        loss = None
        for epoch in range(self.minimizing_epochs):
            self.opt.zero_grad()
            loss = -torch.mean(weights * self.log_likelihood(feat, actions))
            loss.backward()
            self.opt.step()

        loglike_after = self.log_likelihood(feat, actions)
        return {
            "policy_loss": loss,
            "kl_loss": F.kl_div(
                loglike_before, loglike_after, log_target=True, reduction="batchmean"
            ),
        }

    def log_likelihood(self, feat, taken_actions) -> torch.FloatTensor:
        return self._dist(feat).log_prob(taken_actions)

    def set_eval_mode(self, enabled):
        self.stochastic = not enabled
