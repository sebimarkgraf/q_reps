import logging
from typing import Callable, Union

import dm_env
import numpy as np
import torch
import torch.nn as nn
from bsuite.baselines import base
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from .buffer import ReplayBuffer
from .policy import Policy
from .util import center_advantages, to_torch

logger = logging.getLogger("reps")
logger.addHandler(logging.NullHandler())


class REPS(base.Agent, nn.Module):
    """Feed-forward actor-critic agent."""

    def __init__(
        self,
        buffer_size: int,
        policy: Policy,
        value_function,
        epsilon=1e-5,
        writer: SummaryWriter = None,
        center_advantages=False,
        dual_opt_steps=500,
        pol_opt_steps=300,
        batch_size=1000,
        gamma=0.99,
        lr=1e-2,
        optimizer=torch.optim.Adam,
    ):
        super().__init__()

        # Setup for dual
        self.log_eta = nn.Parameter(torch.zeros(1,))
        self.epsilon = torch.tensor(epsilon)
        self.value_function = value_function
        self.dual_opt_steps = dual_opt_steps
        self.gamma = gamma

        # Policy Setup
        self.policy = policy
        self.pol_opt_steps = pol_opt_steps

        self.center_advantages = center_advantages
        self.buffer = ReplayBuffer(buffer_size)
        self.writer = writer
        self.batch_size = batch_size
        self.optimizer = optimizer(self.parameters(), lr=lr)

    def dual(
        self, features: Tensor, features_next: Tensor, rewards: Tensor, _actions: Tensor
    ):
        """
        Implements REPS loss

        @param features: the batched features of the state
        @param features_next: the batches features for the next state i.e. features[1:]
        @param rewards: the batches rewards for the transitions
        @param _actions: action for having the same signature in dual and nll_loss
        @return: the calculated dual function value, supporting autograd of PyTorch
        """
        value = self.value_function(features)
        weights = self.calc_weights(features, features_next, rewards)
        normalizer = torch.logsumexp(weights, dim=0)
        dual = self.eta() * (self.epsilon + normalizer) + (1.0 - self.gamma) * value
        # + l2_reg_dual * (torch.square(self.eta) + torch.square(1 / self.eta))

        return dual.mean(0)

    def select_action(self, timestep: dm_env.TimeStep) -> Union[int, np.array]:
        """Selects actions using current policy in an on-policy setting"""
        obs_feat = torch.tensor([timestep.observation]).float()
        action = self.policy.sample(obs_feat)
        return action

    def optimize_loss(self, loss_fn: Callable, opt_steps):
        loss = 0
        for _ in range(opt_steps):
            (
                next_observations,
                actions,
                rewards,
                discounts,
                observations,
            ) = self.buffer.sample(self.batch_size)
            self.optimizer.zero_grad()
            loss = loss_fn(observations, next_observations, rewards, actions)
            loss.backward()
            self.optimizer.step()

        return loss

    def bellman_error(self, features, features_next, rewards):
        value = self.value_function(features)
        value_next = self.value_function(features_next)
        value_target = rewards + self.gamma * value_next
        return value_target - value

    def eta(self):
        """Eta as a function from the logarithm. Forces eta to be positive"""
        return torch.exp(self.log_eta)

    def calc_weights(
        self, features: Tensor, features_next: Tensor, rewards: Tensor
    ) -> Tensor:
        """
        Calculate the weights from the advantage for updating the policy

        @param features: batched features for the states [N, feature_dim]
        @param features_next: batches features for the following states (e.g. features[1:]) [N, feature_dim]
        @param rewards: undiscounted rewards received in the states [N]
        @return: Tuple of the weights, calculated advantages
        """
        advantage = self.bellman_error(features, features_next, rewards)

        if self.center_advantages:
            advantage = center_advantages(advantage)

        weights = advantage / self.eta()
        return weights

    def update_policy(self, iteration):

        (
            next_observations,
            actions,
            rewards,
            discounts,
            observations,
        ) = self.buffer.sample(self.batch_size)
        dual_loss = self.dual(observations, next_observations, rewards, actions)
        pol_loss = self.nll_loss(observations, next_observations, rewards, actions)
        dist_before = self.policy._dist(observations)

        logger.info(
            f"Iteration {iteration} Before, Sum reward: {torch.sum(rewards):.2f}, Dual Loss: {dual_loss.item():.2f}, "
            f"Policy Loss {pol_loss.item():.2f}"
        )

        rewards = to_torch(rewards)
        actions = to_torch(actions)

        dual_loss = self.optimize_loss(self.dual, self.dual_opt_steps)
        pol_loss = self.optimize_loss(self.nll_loss, self.pol_opt_steps)

        self.buffer.reset()

        dual_loss = self.dual(observations, next_observations, rewards, actions)
        pol_loss = self.nll_loss(observations, next_observations, rewards, actions)
        dist_after = self.policy._dist(observations)
        kl_loss = torch.distributions.kl_divergence(dist_before, dist_after).mean(0)

        logger.info(
            f"Iteration {iteration} After, Sum reward: {torch.sum(rewards):.2f}, Dual Loss: {dual_loss.item():.2f}, "
            f"Policy Loss {pol_loss.item():.2f}, "
            f"KL Loss {kl_loss.item():.2f}"
        )
        if self.writer is not None:
            self.writer.add_scalar("train/pol_loss", pol_loss, iteration)
            self.writer.add_scalar("train/reward", torch.sum(rewards), iteration)
            self.writer.add_scalar("train/dual_loss", dual_loss, iteration)
            self.writer.add_histogram("train/actions", actions, iteration)
            self.writer.add_scalar("train/kl_loss", kl_loss, iteration)

    def nll_loss(self, observations, next_observations, rewards, actions):
        weights = self.calc_weights(observations, next_observations, rewards)
        nll = weights.detach() * self.policy.log_likelihood(observations, actions)
        return -torch.mean(torch.clamp_max(nll, 1e-3))

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: base.Action,
        new_timestep: dm_env.TimeStep,
    ):
        self.buffer.push(new_timestep, action, new_timestep)
