import logging
from typing import Callable, Tuple, Union

import dm_env
import nlopt
import numpy as np
import torch
from bsuite.baselines import base
from bsuite.baselines.utils import sequence
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from .advantages import center_advantages
from .buffer import Buffer
from .policy import Policy
from .reps_dual import reps_dual
from .util import to_torch

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
        obs_feat = self.pol_feature_fn([timestep.observation])[0]
        action = self.policy.sample(obs_feat)
        return action

    def value_function(self, features: Tensor):
        return self.theta.dot(features)

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

    def calc_weights(self, features, rewards):
        advantage = rewards - self.theta.matmul(features.T)
        if self.center_advantages:
            advantage = center_advantages(advantage)
        return torch.exp(
            torch.clamp(advantage / self.eta, -self.exp_limit, self.exp_limit)
        )

    def update_policy(self, trajectory: sequence.Trajectory):
        observations, actions, rewards, discounts = trajectory
        features = self.feature_fn(observations[:-1])
        features_next = self.feature_fn(observations[1:])
        rewards = to_torch(rewards)
        discounted_rewards = to_torch(rewards * discounts)
        actions = to_torch(actions)

        dual_loss = self.optimize_dual(features, features_next, discounted_rewards)

        # Calculate weights
        weights = self.calc_weights(features, rewards)

        pol_features = self.pol_feature_fn(observations[:-1])
        policy_loss = self.policy.fit(pol_features, actions, weights)

        logger.info(
            f"Mean reward: {torch.mean(rewards):.2f}, Dual Loss: {dual_loss:.2f}"
        )
        if self.writer is not None:
            for key, value in policy_loss.items():
                self.writer.add_scalar("train/" + key, value, self.iter)
            self.writer.add_scalar("train/mean_reward", torch.mean(rewards), self.iter)
            self.writer.add_scalar("train/dual_loss", dual_loss, self.iter)
            self.writer.add_histogram("train/actions", actions, self.iter)

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: base.Action,
        new_timestep: dm_env.TimeStep,
    ):
        """Sampling: Obtain N samples (s_i, a_i, s_i', r_i)

        Relevant from experiment
        for _ in range(num_episodes):
        # Run an episode.
        timestep = environment.reset()
        while not timestep.last():
          # Generate an action from the agent's policy.
          action = agent.select_action(timestep)

          # Step the environment.
          new_timestep = environment.step(action)

          # Tell the agent about what just happened.
          agent.update(timestep, action, new_timestep)

          # Book-keeping.
          timestep = new_timestep

        """
        self.buffer.append(timestep, action, new_timestep)
        if self.buffer.full() or new_timestep.last():
            trajectory = self.buffer.drain()
            self.update_policy(trajectory)
            self.iter += 1
