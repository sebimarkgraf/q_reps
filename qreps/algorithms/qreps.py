import logging
import math
from collections import Callable

import dm_env
import torch
from bsuite.baselines import base
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from qreps.algorithms.abstract_algorithm import AbstractAlgorithm
from qreps.algorithms.sampler.exponentited_gradient import ExponentitedGradientSampler
from qreps.memory.replay_buffer import ReplayBuffer
from qreps.policies.stochasticpolicy import StochasticPolicy
from qreps.utilities.elbe import empirical_bellman_error
from qreps.valuefunctions.integrated_q_function import IntegratedQFunction
from qreps.valuefunctions.q_function import SimpleQFunction

logger = logging.getLogger("reps")
logger.addHandler(logging.NullHandler())


class QREPS(AbstractAlgorithm):
    """Feed-forward actor-critic agent."""

    def __init__(
        self,
        q_function: SimpleQFunction,
        policy: StochasticPolicy,
        saddle_point_steps=300,
        beta=0.1,
        eta=0.5,
        learner=torch.optim.SGD,
        buffer_size=10000,
        sampler=ExponentitedGradientSampler,
        sampler_args={},
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.buffer = ReplayBuffer(buffer_size)
        self.policy = policy
        self.saddle_point_steps = saddle_point_steps
        self.eta = eta
        self.q_function = q_function
        self.value_function = IntegratedQFunction(self.policy, self.q_function)
        self.theta_opt = learner(self.q_function.parameters(), lr=beta)
        self.pol_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-2)
        self.optimize_policy = True
        self.sampler = sampler
        self.sampler_args = sampler_args

        # Setting alpha to eta, as mentioned in Paper page 19
        self.alpha = eta

    def g_hat(self, x_1, a_1, x, a):
        return self.discount * self.q_function.features(
            x_1, a_1
        ) - self.q_function.features(x, a)

    def theta_policy(self, x):
        distribution = self.policy.distribution(x)
        value = self.value_function(x)

        def func(a):
            return self.alpha * (self.q_function(x, a) - value)

        if not distribution.has_enumerate_support:
            raise Exception("Not supported distribution for QREPS")

        actions = []
        values = []
        for action in distribution.enumerate_support():
            q_values = func(action)
            log_probs = distribution.log_prob(action)
            value = q_values * torch.exp(log_probs.detach())
            values.append(value)
            actions.append(action)

        dist = torch.distributions.Categorical(logits=torch.tensor(values))
        sample = dist.sample((1,))

        return actions[sample]

    def qreps_eval(self, features, features_next, actions, rewards):
        N = features.shape[0]
        # Initialize z as uniform distribution over all samples
        z = torch.ones((N,))
        z = z / torch.sum(z)
        sampler = self.sampler(length=N, eta=self.eta, **self.sampler_args)

        # Keep history of parameters for averaging
        # If changing to other functions than linear as features, this should be changes to take all parameters
        theta_hist = torch.zeros(
            (self.saddle_point_steps,) + self.q_function.model.weight.size()
        )
        theta_hist[0] = self.q_function.model.weight

        for tau in range(1, self.saddle_point_steps):
            # Learner
            self.theta_opt.zero_grad()
            z_dist = torch.distributions.Categorical(z)
            sample_index = z_dist.sample((1,)).item()
            x, a = features[sample_index].view(1, -1), actions[sample_index].view(1, -1)
            x1 = features_next[sample_index].view(1, -1)
            a1 = self.theta_policy(x1).view(1, -1)
            self.q_function.model.weight.backward(self.g_hat(x1, a1, x, a).squeeze(1))
            self.theta_opt.step()
            theta_hist[tau] = self.q_function.model.weight

            # Sampler
            with torch.no_grad():
                bellman = empirical_bellman_error(
                    features,
                    features_next,
                    actions,
                    rewards,
                    self.q_function,
                    self.value_function,
                    self.discount,
                )
                z = sampler.get_next_distribution(bellman)
            if torch.isnan(z).any() or torch.isnan(bellman).any():
                print(
                    f"In Iteration {tau} contains NaN:\n"
                    f"Z: {torch.isnan(z).any()}, "
                    f"bellman: {torch.isnan(bellman).any()} "
                )
                raise RuntimeError("Ran into nan")

        # Average over the weights
        with torch.no_grad():
            self.q_function.model.weight.data = torch.mean(theta_hist, 0)

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
        loss = torch.sum(torch.exp(z) * (bellman_error - (math.log(N) + z) / self.eta))
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
        if observations.ndim < 2:
            observations = observations.view(-1, 1)
        if next_observations.ndim < 2:
            next_observations = next_observations.view(-1, 1)
        actions = actions.view(-1, 1)
        rewards = self.get_rewards(rewards)
        dist_before = self.policy.distribution(observations)

        qreps_loss = self.qreps_eval(observations, next_observations, actions, rewards)

        if self.optimize_policy is True:
            self.optimize_loss(self.nll_loss, self.pol_optimizer)

        self.buffer.reset()

        elbe_loss = (
            1
            / self.eta
            * torch.logsumexp(
                empirical_bellman_error(
                    observations,
                    next_observations,
                    actions,
                    rewards,
                    self.q_function,
                    self.value_function,
                    self.discount,
                ),
                0,
            )
        )
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
        self, loss_fn: Callable, optimizer: torch.optim.Optimizer, optimizer_steps=50
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
        if len(observations.shape) < 2:
            observations = observations.view(-1, 1)
        if len(next_observations.shape) < 2:
            next_observations = next_observations.view(-1, 1)
        actions = actions.view(-1, 1)
        rewards = self.get_rewards(rewards)

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
