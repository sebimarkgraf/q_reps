import logging

import torch
import torch.nn as nn
from torch import Tensor

from qreps.policies import DirectSetPolicy
from qreps.valuefunctions import AbstractValueFunction

from .abstract_algorithm import AbstractAlgorithm

logger = logging.getLogger("reps")
logger.addHandler(logging.NullHandler())


class REPS(AbstractAlgorithm):
    """Feed-forward actor-critic agent."""

    def __init__(
        self,
        value_function: AbstractValueFunction,
        dual_opt_steps=150,
        batch_size=1000,
        optimize_policy=True,
        entropy_constrained=True,
        eta=0.5,  # Default from Q-REPS paper
        dual_optimizer=torch.optim.Adam,
        dual_lr=1e-2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Setup for dual
        if entropy_constrained:
            # Original formulation of REPS with constrained entropy
            self.log_eta = nn.Parameter(torch.tensor(0.0))
            self.epsilon = torch.tensor(eta)
        else:
            # Formulation of REPS as mirror descent with entropy regularization
            self.log_eta = torch.log(torch.tensor(eta))
            self.epsilon = torch.tensor(0.0)

        self.value_function = value_function
        self.dual_opt_steps = dual_opt_steps

        self.batch_size = batch_size
        self.dual_optimizer = dual_optimizer(
            self.value_function.parameters(), lr=dual_lr
        )
        self.optimize_policy = optimize_policy

    def dual(
        self, features: Tensor, features_next: Tensor, rewards: Tensor, actions: Tensor
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
        weights = self.calc_weights(features, features_next, rewards, actions)
        normalizer = torch.logsumexp(weights, dim=0)
        dual = self.eta() * (self.epsilon + normalizer) + (1.0 - self.discount) * value

        return dual.mean(0)

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
        value_target = rewards + self.discount * value_next
        return value_target - value

    def eta(self):
        """Eta as a function from the logarithm. Forces eta to be positive"""
        return torch.exp(self.log_eta) + float(1e-6)

    def calc_weights(
        self, features: Tensor, features_next: Tensor, rewards: Tensor, actions: Tensor
    ) -> Tensor:
        """
        Calculate the weights from the advantage for updating the policy

        @param features: batched features for the states [N, feature_dim]
        @param features_next: batches features for the following states (e.g. features[1:]) [N, feature_dim]
        @param rewards: undiscounted rewards received in the states [N]
        @return: Tuple of the weights, calculated advantages
        """
        advantage = self.bellman_error(
            features, features_next, self.get_rewards(rewards)
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
        rewards = self.get_rewards(rewards)

        self.optimize_loss(self.dual, self.dual_optimizer, optimizer_steps=300)

        if isinstance(self.policy, DirectSetPolicy):
            self.optimize_policy = False
            self.policy.set_likelihoods(
                observations,
                actions,
                self.calc_weights(observations, next_observations, rewards),
            )

        if self.optimize_policy is True:
            self.optimize_loss(
                self.nll_loss, self.pol_optimizer, optimizer_steps=self.policy_opt_steps
            )

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
            self.writer.add_scalar(
                "train/values",
                self.value_function(next_observations).mean(0),
                iteration,
            )
