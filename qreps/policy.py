from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch


class Policy(ABC):
    @abstractmethod
    def sample(self, observation: torch.Tensor) -> Union[int, np.array]:
        """Sample the policy"""
        pass

    @abstractmethod
    def fit(
        self, feat: torch.Tensor, actions: torch.Tensor, weights: torch.Tensor
    ) -> torch.FloatTensor:
        """Fit the policy to the provided samples"""
        pass

    def set_eval_mode(self, enabled: bool):
        pass


class DiscreteStochasticPolicy(Policy):
    def __init__(self, n_states: int, n_actions: int):
        self._policy = torch.ones((n_states, n_actions))
        self._policy /= torch.sum(self._policy, 1, keepdim=True)

    def sample(self, observation):
        """Expect observation to just be the state"""
        m = torch.distributions.Categorical(self._policy[observation])
        return int(m.sample())

    def fit(self, feats, actions, weights):
        states = feats

        for s, a, w in zip(states.long().numpy(), actions.long().numpy(), weights):
            self._policy[s, a] = self._policy[s, a] * w

        self._policy = torch.softmax(self._policy, 1)

        # FIXME: Return real loss
        return 0


class CategoricalMLP(Policy, torch.nn.Module):
    def __init__(self, obs_shape, act_shape, minimizing_epochs=100, lr=1e-2):
        super(CategoricalMLP, self).__init__()
        self.minimizing_epochs = minimizing_epochs
        self.layer = torch.nn.Linear(obs_shape, act_shape, bias=False)

        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.stochastic = True

    def _dist(self, observation) -> torch.distributions.Distribution:
        return torch.distributions.categorical.Categorical(
            logits=self.layer(observation)
        )

    @torch.no_grad()
    def sample(self, observation):
        if self.stochastic:
            return self._dist(observation).sample().item()
        else:
            return torch.argmax(self.layer(observation))

    def fit(self, feat, actions, weights):

        loss = None
        for epoch in range(self.minimizing_epochs):
            self.opt.zero_grad()
            loss = self.log_likelihood(actions, feat, weights)
            loss.backward()
            self.opt.step()
        return loss

    def log_likelihood(self, taken_actions, feat, weights) -> torch.FloatTensor:
        log_lik = self._dist(feat).log_prob(taken_actions)
        return -torch.mean(weights * log_lik)

    def set_eval_mode(self, enabled):
        self.stochastic = not enabled


class GaussianMLP(Policy, torch.nn.Module):
    def __init__(self, obs_shape, act_shape, minimizing_epochs=300, lr=1e-2):
        super(GaussianMLP, self).__init__()
        self.minimizing_epochs = minimizing_epochs
        self._mu = torch.nn.Linear(obs_shape, act_shape, bias=False)
        self._sigma = torch.nn.Parameter(torch.ones(act_shape))

        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.stochastic = True

    def _dist(self, observation) -> torch.distributions.Distribution:
        return torch.distributions.normal.Normal(self._mu(observation), self._sigma)

    @torch.no_grad()
    def sample(self, observation):
        if self.stochastic:
            return self._dist(observation).sample()
        else:
            return self._mu(observation)

    def fit(self, feat, actions, weights):
        loss = None
        self._mu.reset_parameters()
        for epoch in range(self.minimizing_epochs):
            self.opt.zero_grad()
            loss = self.log_likelihood(actions, feat, weights)
            loss.backward()
            self.opt.step()
        return loss

    def log_likelihood(self, taken_actions, feat, weights) -> torch.FloatTensor:
        log_lik = self._dist(feat).log_prob(taken_actions)
        return -torch.mean(weights * log_lik)

    def set_eval_mode(self, enabled):
        self.stochastic = not enabled
