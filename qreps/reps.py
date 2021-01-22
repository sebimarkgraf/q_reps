import logging
from typing import Callable, Tuple, Union

import dm_env
import nlopt
import numpy as np
import torch
import torch.nn.functional as F
from bsuite.baselines import base
from bsuite.baselines.utils import sequence
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from .buffer import Buffer
from .feature_functions import bellman_error_batched
from .policy import Policy
from .reps_dual import reps_dual
from .util import center_advantages, to_torch

logger = logging.getLogger("reps")
logger.addHandler(logging.NullHandler())


class REPS(base.Agent):
    """Feed-forward actor-critic agent."""

    def __init__(
        self,
        feat_shape: Tuple,
        sequence_length: int,
        policy: Policy,
        val_feature_fn: Callable[[np.array], Tensor],
        pol_feature_fn: Callable[[np.array], Tensor],
        epsilon=1e-5,
        dual_optimizer_algorithm=nlopt.LD_SLSQP,
        writer: SummaryWriter = None,
        center_advantages=True,
        exp_limit=200.0,
        l2_reg_dual=0,
    ):
        logger.info(f"Observations: {feat_shape}")
        logger.info(
            f"Using Optimizer: {nlopt.algorithm_name(dual_optimizer_algorithm)}"
        )
        self.eta = torch.rand((1,))
        self.epsilon = torch.tensor(epsilon)
        self.buffer = Buffer(sequence_length)
        self.feature_fn = val_feature_fn
        self.pol_feature_fn = pol_feature_fn
        self.theta = torch.rand(feat_shape)
        self.dual_optimizer_algorithm = dual_optimizer_algorithm
        self.center_advantages = center_advantages
        self.exp_limit = exp_limit
        self.l2_reg_dual = l2_reg_dual
        self.policy = policy
        self.stochastic = True
        self.writer = writer
        self.iter = 0

    def select_action(self, timestep: dm_env.TimeStep) -> Union[int, np.array]:
        """Selects actions using current policy in an on-policy setting"""
        obs_feat = self.pol_feature_fn([timestep.observation])
        action = self.policy.sample(obs_feat)
        return action

    def optimize_dual(self, features: Tensor, features_next: Tensor, rewards: Tensor):
        def eval_fn(x: np.array, grad: np.array):
            eta = torch.tensor(
                x[0], dtype=torch.get_default_dtype(), requires_grad=True
            )
            theta = torch.tensor(
                x[1:], dtype=torch.get_default_dtype(), requires_grad=True
            )
            dual_val = reps_dual(
                eta,
                theta,
                features,
                features_next,
                rewards,
                self.epsilon,
                l2_reg_dual=self.l2_reg_dual,
            )
            dual_val.backward()

            if grad.size > 0:
                grad[0] = eta.grad.numpy()
                grad[1:] = theta.grad.numpy()

            return dual_val.item()

        # SLSQP seems to work but L-BFGS fails without useful errors
        optimizer = nlopt.opt(self.dual_optimizer_algorithm, 1 + len(self.theta))
        optimizer.set_lower_bounds([1e-5] + [-np.inf] * len(self.theta))
        optimizer.set_upper_bounds([1e8] + [np.inf] * len(self.theta))
        optimizer.set_min_objective(eval_fn)
        optimizer.set_ftol_abs(1e-7)
        optimizer.set_maxeval(3000)

        x0 = [1] + [1] * self.theta.shape[0]

        try:
            params_best = optimizer.optimize(x0)
        except:  # noqa:
            params_best = x0
            logger.warning("Stopped optimization due to error.")

        logger.debug(f"New Params: {params_best}")
        logger.debug(f"Optimized dual value: {optimizer.last_optimum_value()}")
        if optimizer.last_optimize_result() == nlopt.MAXEVAL_REACHED:
            logger.info("Stopped due to max eval iterations.")

        # Update params from optimum
        self.eta = to_torch(params_best[0])
        self.theta = to_torch(params_best[1:])

        return optimizer.last_optimum_value()

    def calc_weights(
        self, features: Tensor, features_next: Tensor, rewards: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Calculate the weights from the advantage for updating the policy

        @param features: batched features for the states [N, feature_dim]
        @param features_next: batches features for the following states (e.g. features[1:]) [N, feature_dim]
        @param rewards: undiscounted rewards received in the states [N]
        @return: Tuple of the weights, calculated advantages
        """
        advantage = bellman_error_batched(self.theta, features, features_next, rewards)

        if self.center_advantages:
            advantage = center_advantages(advantage)

        weights = torch.exp(
            torch.clamp(
                advantage / self.eta - torch.max(advantage / self.eta),
                -self.exp_limit,
                self.exp_limit,
            )
        )
        return weights, advantage

    def update_policy(self, trajectory: sequence.Trajectory):
        observations, actions, rewards, discounts = trajectory
        features = self.feature_fn(observations[:-1])
        features_next = self.feature_fn(observations[1:])
        rewards = to_torch(rewards)
        actions = to_torch(actions)

        dual_loss = self.optimize_dual(features, features_next, rewards)

        # Calculate weights
        weights, advantage = self.calc_weights(features, features_next, rewards)

        # The policy could have other features than the value function
        # Need to calculate of all features besides the last -> Last observation has no action
        pol_features = self.pol_feature_fn(observations[:-1])
        # Update the policy according to the calculated weights
        # Returns a dict of all calculated measurements
        policy_loss = self.policy.fit(pol_features, actions, weights)

        # FIXME: Remove this when no longer needed
        # Allows to debug a discrete problem
        # for s, a, adv, w, f in zip(
        #     pol_features.long().numpy(),
        #     actions.long().numpy(),
        #     advantage,
        #     weights,
        #     features,
        # ):
        #     print(
        #         f"State: {s}, Action: {a}, Advantage: {adv:2.2f}, Weight: {w:2.2f}, Values: {self.theta.dot(f)}"
        #     )
        # for s in range(len(self.theta)):
        #    print(f"Value for State {s}: {self.theta.dot(self.feature_fn(s))}")

        logger.info(f"Sum reward: {torch.sum(rewards):.2f}, Dual Loss: {dual_loss:.2f}")
        if self.writer is not None:
            for key, value in policy_loss.items():
                self.writer.add_scalar("train/" + key, value, self.iter)
            self.writer.add_scalar("train/reward", torch.sum(rewards), self.iter)
            self.writer.add_scalar("train/dual_loss", dual_loss, self.iter)
            self.writer.add_histogram("train/actions", actions, self.iter)

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: base.Action,
        new_timestep: dm_env.TimeStep,
    ):
        """Sampling: Obtain N samples (s_i, a_i, s_i', r_i)

        Currently done using the Agent interface of DM.
        TODO: Should be moved to a Trainer class in the future.
        """
        self.buffer.append(timestep, action, new_timestep)
        if self.buffer.full() or new_timestep.last():
            trajectory = self.buffer.drain()
            self.update_policy(trajectory)
            self.iter += 1
