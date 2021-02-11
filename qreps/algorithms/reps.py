import logging
from typing import Callable, Union

import dm_env
import numpy as np
import torch
import torch.nn as nn
from bsuite.baselines import base
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from qreps.algorithms.abstract_algorithm import AbstractAlgorithm
from qreps.memory.replay_buffer import ReplayBuffer
from qreps.policies.direct_set_policy import DirectSetPolicy
from qreps.policies.stochasticpolicy import StochasticPolicy

logger = logging.getLogger("reps")
logger.addHandler(logging.NullHandler())


class REPS(AbstractAlgorithm):
    """Feed-forward actor-critic agent."""

    def __init__(
        self,
        buffer_size: int,
        policy: StochasticPolicy,
        value_function,
        dual_opt_steps=500,
        pol_opt_steps=300,
        batch_size=1000,
        gamma=1.0,
        lr=1e-3,
        optimizer=torch.optim.Adam,
        optimize_policy=True,
        entropy_constrained=True,
        eta=0.5,  # Default from Q-REPS paper
        dual_optimizer=torch.optim.Adam,
        dual_lr=1e-2,
        reward_transformer=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Setup for dual
        if entropy_constrained:
            # Original formulation of REPS with constrained entropy
            self.log_eta = nn.Parameter(torch.log(torch.tensor(1.0)))
            self.epsilon = torch.tensor(eta)
        else:
            # Formulation of REPS as mirror descent with entropy regularization
            self.log_eta = torch.log(torch.tensor(eta))
            self.epsilon = torch.tensor(0.0)

        self.value_function = value_function
        self.dual_opt_steps = dual_opt_steps
        self.gamma = gamma

        # Policy Setup
        self.policy = policy
        self.pol_opt_steps = pol_opt_steps

        self.batch_size = batch_size
        self.pol_optimizer = optimizer(self.policy.parameters(), lr=lr)
        self.dual_optimizer = dual_optimizer(
            self.value_function.parameters(), lr=dual_lr
        )
        self.optimize_policy = optimize_policy
        self.reward_transformer = reward_transformer

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
        # value = self.value_function(features)
        weights = self.calc_weights(features, features_next, rewards)
        normalizer = torch.logsumexp(weights, dim=0)
        dual = self.eta() * (self.epsilon + normalizer)  # + (1.0 - self.gamma) * value

        return dual.mean(0)

    def select_action(self, timestep: dm_env.TimeStep) -> Union[int, np.array]:
        """
        Selects actions using current policy in an on-policy setting.

        @param timestep: the current timestep containg the observation
        @return: An action confirming to the dm_control numpy or int convention
        """
        obs_feat = torch.tensor(timestep.observation).float()
        action = self.policy.sample(obs_feat)
        return action

    def optimize_loss(
        self, loss_fn: Callable, optimizer: torch.optim.Optimizer, optimizer_steps=300
    ):
        """
        Optimize the specified loss using batch gradient descent.

        Allows to specify an optimizer and is compatible with L-BFGS, Adam and SGD.

        @param loss_fn: the loss function which should be minimized.
        @param optimizer: the torch optimizer to use
        @param optimizer_steps: how many steps to do the optimization
        """
        (
            next_observations,
            actions,
            rewards,
            discounts,
            observations,
        ) = self.buffer.get_all()

        # This is implemented using a closure mainly due to the potential usage of BFGS
        # BFGS needs to evaluate the function multiple times and therefore needs a defined closure
        # All other optimizers handle the closure just fine as well, but only execute it once
        def closure():
            optimizer.zero_grad()
            loss = loss_fn(observations, next_observations, rewards, actions)
            loss.backward()
            return loss

        for i in range(optimizer_steps):
            optimizer.step(closure)

    def bellman_error(self, features, features_next, rewards):
        """
        Calculates the bellman or TD-error as difference between current and next state.

        The calculation respects the specified gamma. The implementation works with batches.
        @param features: the features of the current state
        @param features_next:  the features of the next state
        @param rewards: the received rewards.
        @return:
        """
        value = self.value_function(features)
        value_next = self.value_function(features_next)
        value_target = rewards + self.gamma * value_next
        return value_target - value

    def eta(self):
        """Eta as a function from the logarithm. Forces eta to be positive"""
        return torch.exp(self.log_eta) + float(1e-6)

    def get_normalized_rewards(self, rewards):
        """
        Get the transformed rewards for the given rewards.

        Applies the specified transformers.
        @param rewards: rewards to transform
        @return: the transformed rewards
        """
        if self.reward_transformer is not None:
            return self.reward_transformer(rewards)
        else:
            return rewards

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
        advantage = self.bellman_error(
            features, features_next, self.get_normalized_rewards(rewards)
        )
        weights = advantage / self.eta()
        return weights

    def update_policy(self, iteration):

        (
            next_observations,
            actions,
            rewards,
            discounts,
            observations,
        ) = self.buffer.get_all()
        dist_before = self.policy.distribution(observations)

        self.optimize_loss(self.dual, self.dual_optimizer, optimizer_steps=300)

        if isinstance(self.policy, DirectSetPolicy):
            self.optimize_policy = False
            self.policy.set_likelihoods(
                observations,
                actions,
                self.calc_weights(observations, next_observations, rewards),
            )

        if self.optimize_policy is True:
            self.optimize_loss(self.nll_loss, self.pol_optimizer)

        self.buffer.reset()
        dual_loss = self.dual(observations, next_observations, rewards, actions)
        dist_after = self.policy.distribution(observations)

        self.report_tensorboard(
            observations,
            next_observations,
            rewards,
            actions,
            dist_before,
            dist_after,
            iteration,
        )
        if self.writer is not None:
            self.writer.add_scalar("train/dual_loss", dual_loss, iteration)

        logger.info(
            f"Iteration {iteration} After, Sum reward: {torch.sum(rewards):.2f}, Dual Loss: {dual_loss.item():.2f}"
        )

    def nll_loss(self, observations, next_observations, rewards, actions):
        weights = self.calc_weights(observations, next_observations, rewards)
        log_likes = self.policy.log_likelihood(observations, actions)
        nll = weights.detach() * log_likes
        return -torch.mean(torch.clamp_max(nll, 1e-3))

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: base.Action,
        new_timestep: dm_env.TimeStep,
    ):
        self.buffer.push(timestep, action, new_timestep)

    def save(self, filename: str):
        return torch.save(self, filename)
