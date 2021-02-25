import logging
import math

import torch
from torch import Tensor
from typing_extensions import Type

from qreps.algorithms.sampler import AbstractSampler, ExponentitedGradientSampler
from qreps.policies import StochasticPolicy
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
        policy: StochasticPolicy,
        saddle_point_steps: int = 300,
        beta: float = 0.1,
        eta: float = 0.5,
        learner: Type[torch.optim.Optimizer] = torch.optim.SGD,
        sampler: Type[AbstractSampler] = ExponentitedGradientSampler,
        sampler_args: dict = None,
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
        @param args: arguments for the abstract algorithm.
        @param kwargs: keyword arguments for the abstract algorithm.
        """
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.saddle_point_steps = saddle_point_steps
        self.eta = eta
        self.q_function = q_function
        self.value_function = IntegratedQFunction(self.policy, self.q_function)
        self.theta_opt = learner(self.q_function.parameters(), lr=beta)
        self.pol_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.policy_lr
        )
        self.optimize_policy = True
        self.sampler = sampler
        self.sampler_args = sampler_args if sampler_args is not None else {}

        # Setting alpha to eta, as mentioned in Paper page 19
        self.alpha = eta

    def g_hat(
        self, x_1: torch.Tensor, a_1: torch.Tensor, x: torch.Tensor, a: torch.Tensor
    ):
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
            grad_samples = 1
            for i in range(grad_samples):
                sample_index = z_dist.sample((1,)).item()
                x, a = (
                    features[sample_index].view(1, -1),
                    actions[sample_index].view(1, -1),
                )
                x1 = features_next[sample_index].view(1, -1)
                a1 = self.theta_policy(x1).view(1, -1)
                grad += self.g_hat(x1, a1, x, a)
            grad /= grad_samples
            self.q_function.model.weight.backward(grad)

            # indicees = z_dist.sample((10,))
            # loss = self.S_k(z_dist, N, features, features_next, actions, rewards)
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

            self.writer.add_scalar(
                "train/opt_elbe",
                empirical_logistic_bellman(
                    self.eta,
                    features,
                    features_next,
                    actions,
                    rewards,
                    self.q_function,
                    self.value_function,
                    self.discount,
                ),
                self.saddle_point_steps * iteration + tau,
            )

        # Average over the weights
        # with torch.no_grad():
        #    self.q_function.model.weight.data = torch.mean(theta_hist, 0).detach()

        return self.S_k(z_dist, N, features, features_next, actions, rewards)

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
        loss = torch.sum(z.probs * (bellman_error - (math.log(N) + z.probs) / self.eta))
        # + (1 - self.discount) * self.value_function(features))
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

    def nll_loss(self, observations, next_observations, rewards, actions):
        weights = self.calc_weights(observations, next_observations, rewards, actions)
        log_likes = self.policy.log_likelihood(observations, actions)
        nll = weights.detach() * log_likes
        return -torch.mean(torch.clamp_max(nll, 1e-3))
