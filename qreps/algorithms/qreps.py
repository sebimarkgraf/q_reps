import logging
from collections import Callable
from typing import Union

import dm_env
import numpy as np
import torch
import torch.nn as nn
from bsuite.baselines import base
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from qreps.algorithms.abstract_algorithm import AbstractAlgorithm
from qreps.elbe import empirical_bellman_error
from qreps.memory.replay_buffer import ReplayBuffer
from qreps.policies.stochasticpolicy import StochasticPolicy
from qreps.valuefunctions.integrated_q_function import IntegratedQFunction
from qreps.valuefunctions.q_function import AbstractQFunction

logger = logging.getLogger("reps")
logger.addHandler(logging.NullHandler())


class QREPS(AbstractAlgorithm):
    """Feed-forward actor-critic agent."""

    def __init__(
        self,
        feature_fn,
        num_actions,
        feature_dim,
        q_function: AbstractQFunction,
        policy: StochasticPolicy,
        saddle_point_steps=300,
        writer: SummaryWriter = None,
        discount=1.0,
        beta=0.1,
        beta_2=0.1,
        eta=0.5,
        alpha=0.5,
        learner=torch.optim.SGD,
        buffer_size=10000,
    ):
        super().__init__()
        self.buffer = ReplayBuffer(buffer_size)
        self.policy = policy
        self.stochastic = True
        self.writer = writer
        self.saddle_point_steps = saddle_point_steps
        self.eta = eta
        # self.q_func = q_function
        self.beta_2 = beta_2
        self.discount = discount
        self.theta = nn.Linear(feature_dim, 1, bias=False)
        self.value_function = IntegratedQFunction(
            self.policy, self.q_function, obs_dim=feature_dim
        )
        self.theta_opt = learner(self.theta.parameters(), lr=beta)
        self.pol_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-2)
        self.optimize_policy = True
        self.alpha = alpha
        self.feature_fn = feature_fn
        self.num_actions = num_actions

    def q_function(self, features, actions):
        feat = self.feature_fn(features, actions)
        return self.theta(feat).squeeze()

    def select_action(self, timestep: dm_env.TimeStep) -> Union[int, np.array]:
        """Selects actions using current policy in an on-policy setting"""
        obs_feat = torch.tensor([timestep.observation]).float()
        action = self.policy.sample(obs_feat)
        return action

    def g_hat(self, x_1, a_1, x, a):
        return (
            self.discount * self.feature_fn(x_1, a_1)
            - self.feature_fn(x, a)
            # + (1 - self.discount) * torch.cat((x_bar, a_bar), dim=-1)
        )

    def theta_policy(self, x):
        actions = torch.arange(self.num_actions).float()
        state_actions = torch.cartesian_prod(x, actions)
        states, actions = state_actions[:, 0], state_actions[:, 1]
        policy_dist = self.policy.distribution(states).log_prob(actions)
        q_values = (
            policy_dist
            * self.alpha
            * (self.q_function(states, actions) - self.value_function(states))
        )
        dist = torch.distributions.Categorical(logits=q_values)
        return dist.sample((1,))

    def qreps_eval(self, features, features_next, actions, rewards):
        N = features.shape[0]
        h = torch.ones((self.saddle_point_steps, N))
        # Initialize z as uniform distribution over all samples
        z = torch.ones((self.saddle_point_steps, N))
        z /= torch.sum(z, dim=1, keepdim=True)

        # Keep history of parameters for averaging
        # If changing to other functions than linear as features, this should be changes to take all parameters
        theta_hist = torch.zeros((self.saddle_point_steps,) + self.theta.weight.size())
        theta_hist[0] = self.theta.weight

        for tau in range(1, self.saddle_point_steps):
            # Learner
            self.theta_opt.zero_grad()
            # s_loss = self.S_k(z[tau - 1].detach(), N, features, features_next, actions, rewards)
            # s_loss.backward()
            sample_index = torch.multinomial(z[tau], 1)
            x, a = features[sample_index], actions[sample_index]
            x1 = features_next[sample_index]
            a1 = self.theta_policy(x1)
            self.theta.weight.backward(self.g_hat(x1, a1, x, a))
            self.theta_opt.step()
            theta_hist[tau] = self.theta.weight

            # Sampler
            with torch.no_grad():
                z[tau] = z[tau - 1] * torch.exp(self.beta_2 * h[tau - 1])
                z[tau] /= torch.sum(z[tau])

                bellman = empirical_bellman_error(
                    features,
                    features_next,
                    actions,
                    rewards,
                    self.q_function,
                    self.value_function,
                    self.discount,
                )
                h[tau] = bellman - 1 / self.eta * torch.log(N * z[tau])

        # Average over the weights
        with torch.no_grad():
            self.theta.weight.data = torch.mean(theta_hist, 0)

        return self.S_k(z, N, features, features_next, actions, rewards)

    def S_k(self, z, N, features, features_next, actions, rewards):
        bellman_error = empirical_bellman_error(
            features,
            features_next,
            actions,
            rewards,
            self.q_function,
            self.value_function,
            discount=self.discount,
        )
        loss = torch.sum(z * (bellman_error - torch.log(N * z) / self.eta))
        # + (1 - self.discount) * self.value_function(features))
        return loss

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
        advantages = self.alpha * self.q_function(features, actions)
        return advantages

    def update_policy(self, iteration):
        (
            next_observations,
            actions,
            rewards,
            discounts,
            observations,
        ) = self.buffer.get_all()
        pol_loss = self.nll_loss(observations, next_observations, rewards, actions)
        dist_before = self.policy.distribution(observations)

        logger.info(
            f"Iteration {iteration} Before, Sum reward: {torch.sum(rewards):.2f}"
            f"Policy Loss {pol_loss.item():.2f}"
        )

        qreps_loss = self.qreps_eval(observations, next_observations, actions, rewards)

        if self.optimize_policy is True:
            self.optimize_loss(self.nll_loss, self.pol_optimizer)

        self.buffer.reset()

        elbe_loss = empirical_bellman_error(
            observations,
            next_observations,
            actions,
            rewards,
            self.q_function,
            self.value_function,
            self.discount,
        ).sum(0)
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
            self.writer.add_scalar("train/elbe_loss", elbe_loss, iteration)
            self.writer.add_scalar("train/qreps_loss", qreps_loss, iteration)

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

    def nll_loss(self, observations, next_observations, rewards, actions):
        weights = self.calc_weights(observations, next_observations, rewards, actions)
        log_likes = self.policy.log_likelihood(observations, actions)
        nll = weights.detach() * log_likes
        return -torch.mean(torch.clamp_max(nll, 1e-3))

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: base.Action,
        new_timestep: dm_env.TimeStep,
    ):
        """Sampling: Obtain N samples (s_i, a_i, s_i', r_i)

        Currently done using the Agent interface of DM.
        """
        self.buffer.push(timestep, action, new_timestep)
