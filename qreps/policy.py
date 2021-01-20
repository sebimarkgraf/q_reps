from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F


class Policy(ABC):
    @abstractmethod
    def sample(self, observation: torch.Tensor) -> Union[int, np.array]:
        """Sample the policy"""
        pass

    @abstractmethod
    def fit(
        self, feat: torch.Tensor, actions: torch.Tensor, weights: torch.Tensor
    ) -> dict:
        """Fit the policy to the provided samples"""
        pass

    def set_eval_mode(self, enabled: bool):
        pass


class DiscreteStochasticPolicy(Policy):
    """Discrete Policy which assigns every action in every state a probability."""

    def __init__(self, n_states: int, n_actions: int):
        self._policy = torch.ones((n_states, n_actions))
        self._policy /= torch.sum(self._policy, 1, keepdim=True)
        self._eval = False

    def sample(self, observation):
        """Expect observation to just be the state"""
        if self._eval:
            return int(torch.argmax(self._policy[observation]).item())

        m = self._dist(observation)
        return int(m.sample())

    def _dist(self, observation):
        return torch.distributions.Categorical(self._policy[observation])

    def fit(self, feats, actions, weights):
        log_like_before = self._dist(feats).log_prob(actions)

        for s, a, w in zip(feats.long().numpy(), actions.long().numpy(), weights):
            self._policy[s, a] = self._policy[s, a] * w

        self._policy = self._policy / torch.sum(self._policy, dim=1, keepdim=True)

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
    def __init__(
        self,
        obs_shape,
        act_shape,
        action_max,
        action_min,
        sigma=1.0,
        minimizing_epochs=300,
        lr=1e-2,
    ):
        super(GaussianMLP, self).__init__()
        self.minimizing_epochs = minimizing_epochs
        self._mu = torch.nn.Linear(obs_shape, act_shape, bias=False)
        self._sigma = sigma

        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.stochastic = True
        self.action_max = action_max
        self.action_min = action_min

    def _dist(self, observation) -> torch.distributions.Distribution:
        return torch.distributions.normal.Normal(self._mu(observation), self._sigma)

    @torch.no_grad()
    def sample(self, observation):
        if self.stochastic:
            return torch.clamp(
                self._dist(observation).sample(), self.action_min, self.action_max
            )
        else:
            return torch.clamp(self._mu(observation), self.action_min, self.action_max)

    def fit(self, feat, actions, weights):
        loglike_before = self.log_likelihood(feat, actions)
        loss = None
        self._mu.reset_parameters()
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
