import logging
import math

import torch
from torch import Tensor
from typing_extensions import Type

from qreps.algorithms.sampler import AbstractSampler, ExponentitedGradientSampler
from qreps.utilities.elbe import empirical_bellman_error, empirical_logistic_bellman
from qreps.valuefunctions import IntegratedQFunction, SimpleQFunction

from .abstract_algorithm import AbstractAlgorithm

logger = logging.getLogger("qreps")
logger.addHandler(logging.NullHandler())


class QREPS(AbstractAlgorithm):
    r"""Logistic Q-Learning Algorithm.

        Logistic Q-Learning optimizes a linear program of the optimal policy but uses p = d as trick to naturally
        introduce Q-functions.
        The following linear program is optimized:
        math:: \mathrm{maximize} \langle p, r \rangle - \frac{1}{\eta} D(p || p_0)  - \frac{1}{\alpha} H(d || d_0)
        math:: \text{s.t. }E^T d = \gamma P^T p + ( 1 - \gamma) \vu_0
        math:: \phi^T d = \phi^T p

        Through the dual we naturally introduce Q and V functions.
        The paper shows that the optimal solution for theses solves the linear program as well.

        We can get the optimal solution V and Q, by minimizing empirical logistic bellman error (ELBE)
        math:: \hat G (\theta) = \frac{1}{\eta} \log \left(\frac{1}{N}
            \sum_{n=1}^N e^{\eta \hat \nabla_\theta (\xi_{k,n} }\right) + (1 - \gamma) \langle \vu_0 , V_\theta \rangle

        This algorithm implement the version proposed in the paper as MinMaxQREPS.
        Therefore, this uses a mirror descent algorithm formulated as sampler and learner.

        References
        ----------
        [1] J. Bas-Serrano, S. Curi, A. Krause, and G. Neu, “Logistic $Q$-Learning,”
        arXiv:2010.11151 [cs, stat], Oct. 2020, Accessed: Nov. 11, 2020. [Online].
        Available: http://arxiv.org/abs/2010.11151.
        """

    def __init__(
        self,
        q_function: SimpleQFunction,
        saddle_point_steps: int = 300,
        beta: float = 0.1,
        eta: float = 0.5,
        learner: Type[torch.optim.Optimizer] = torch.optim.SGD,
        sampler: Type[AbstractSampler] = ExponentitedGradientSampler,
        sampler_args: dict = None,
        average_weights: bool = True,
        grad_samples: int = 10,
        *args,
        **kwargs,
    ):
        """
        This creates an QREPS algorithm instance.

        @param q_function: The q_function that should be used. Currently only the SimpleQFunction is supported.
        @param policy: The policy that is used for the problem and should be optimized. Only discrete policies currently
        @param saddle_point_steps: How many optimization steps should be done for the learner/sampler optimization.
        @param beta: Learning rate for the q_function parameters.
        @param eta: The entropy regularization parameter.
        @param learner: Which learner should be used. The default recommendations are SGD and Adam.
        @param sampler: Which sampler to use. This is highly problem dependent.
        @param sampler_args: Args that should be provided when creating the sampler.
        @param average_weights: Enable averaging the parameters after the optimization
        @param args: arguments for the abstract algorithm.
        @param kwargs: keyword arguments for the abstract algorithm.
        """
        super().__init__(*args, **kwargs)
        self.saddle_point_steps = saddle_point_steps
        self.eta = eta
        self.alpha = eta
        self.q_function = q_function
        self.value_function = IntegratedQFunction(
            self.policy, self.q_function, alpha=self.alpha
        )
        self.theta_opt = learner(self.q_function.parameters(), lr=beta)

        self.optimize_policy = True
        self.sampler = sampler
        self.sampler_args = sampler_args if sampler_args is not None else {}

        # Setting alpha to eta, as mentioned in Paper page 19
        self.average_weights = average_weights
        self.grad_samples = grad_samples

    def g_hat(
        self,
        x_1: torch.Tensor,
        a_1: torch.Tensor,
        x: torch.Tensor,
        a: torch.Tensor,
        x0,
        a0,
    ):
        features_1 = self.q_function.features(x_1, a_1)
        features = self.q_function.features(x, a)
        features_0 = self.q_function.features(x0, a0)
        return (self.discount * features_1 - features) + (
            1 - self.discount
        ) * features_0

    def theta_policy(self, x):
        distribution = self.policy.distribution(x)
        value_state = self.value_function(x)

        if not distribution.has_enumerate_support:
            raise Exception("Not supported distribution for QREPS")

        actions = distribution.enumerate_support()
        probs = distribution.probs
        q_values = torch.tensor([self.q_function(x, a) for a in actions])
        policy_values = self.alpha * (q_values - value_state)
        values = torch.exp(policy_values) * probs

        dist = torch.distributions.Categorical(logits=values)
        sample = dist.sample((1,))
        sampled_actions = actions[sample]
        return sampled_actions

    def qreps_eval(self, features, features_next, actions, rewards, iteration):
        N = features.shape[0]
        # Initialize z as uniform distribution over all samples
        sampler = self.sampler(length=N, eta=self.eta, **self.sampler_args)
        z_dist = sampler.get_distribution()

        # Keep history of parameters for averaging
        # If changing to other functions than linear as features, this should be changes to take all parameters
        theta_hist = torch.zeros(
            (self.saddle_point_steps,) + self.q_function.model.weight.size()
        )
        theta_hist[0] = self.q_function.model.weight

        for tau in range(1, self.saddle_point_steps):
            # Learner
            self.theta_opt.zero_grad()
            grad = torch.zeros(self.q_function.model.weight.shape)
            for i in range(self.grad_samples):
                sample_index = z_dist.sample((1,)).item()
                x, a, x1 = (
                    features[sample_index].view(1, -1),
                    actions[sample_index].view(1, -1),
                    features_next[sample_index].view(1, -1),
                )
                a1 = self.theta_policy(x1).view(1, -1)
                x0, a0 = features[0].view(1, -1), actions[0].view(1, -1)
                grad += self.g_hat(x1, a1, x, a, x0, a0)
            grad /= self.grad_samples
            self.q_function.model.weight.backward(grad)

            # loss = self.S_k(z_dist, N, features, features_next, actions, rewards)
            # loss.backward()

            # loss = empirical_logistic_bellman(self.eta, features, features_next, actions, rewards, self.q_function, self.value_function, self.discount)
            # loss.backward()
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
                z_dist = sampler.get_next_distribution(bellman)

        # Average over the weights
        if self.average_weights is True:
            with torch.no_grad():
                self.q_function.model.weight.data = torch.mean(theta_hist, 0).detach()

        return empirical_logistic_bellman(
            self.eta,
            features,
            features_next,
            actions,
            rewards,
            self.q_function,
            self.value_function,
            self.discount,
        )

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
        loss = torch.sum(
            z.probs.detach()
            * (bellman_error - (torch.log(N * z.probs.detach())) / self.eta)
            + (1 - self.discount) * self.value_function(features)
        )
        return loss

    def calc_weights(
        self, features: Tensor, features_next: Tensor, rewards: Tensor, actions: Tensor
    ) -> Tensor:
        """
        Calculate the weights from the advantage for updating the policy

        @param actions: the taken actions
        @param features: batched features for the states [N, feature_dim]
        @param features_next: batches features for the following states (e.g. features[1:]) [N, feature_dim]
        @param rewards: undiscounted rewards received in the states [N]
        @return: Tuple of the weights, calculated advantages
        """
        return self.alpha * self.q_function(features, actions)

    def update_policy(self, iteration):
        (
            next_observations,
            actions,
            rewards,
            discounts,
            observations,
        ) = self.buffer.get_all()

        rewards = self.get_rewards(rewards)
        dist_before = self.policy.distribution(observations)

        qreps_loss = self.qreps_eval(
            observations, next_observations, actions, rewards, iteration
        )

        if self.optimize_policy is True:
            self.optimize_loss(
                self.nll_loss, self.pol_optimizer, optimizer_steps=self.policy_opt_steps
            )

        self.buffer.reset()

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
            self.writer.add_scalar("train/qreps_loss", qreps_loss, iteration)
            self.writer.add_scalar(
                "train/q_values",
                self.q_function(observations, actions).mean(0),
                iteration,
            )
            self.writer.add_scalar(
                "train/next_v_function",
                self.value_function(next_observations).mean(0),
                iteration,
            )
            self.writer.add_histogram(
                "train/q_weights", self.q_function.model.weight.data, iteration
            )
