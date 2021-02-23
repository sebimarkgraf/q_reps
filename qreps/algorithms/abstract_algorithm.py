import logging
from abc import ABCMeta
from typing import Callable, Union

import dm_env
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from qreps.memory.replay_buffer import ReplayBuffer

DEFAULT_REPLAY_BUFFER_SIZE = 100000

logger = logging.getLogger("algorithm")
logger.addHandler(logging.NullHandler())


class AbstractAlgorithm(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        writer: SummaryWriter = None,
        buffer=None,
        reward_transformer=None,
        discount=1.0,
        policy_opt_steps=150,
        policy_lr=1e-2,
    ):
        super().__init__()
        self.writer = writer
        if buffer is None:
            self.buffer = ReplayBuffer(DEFAULT_REPLAY_BUFFER_SIZE)
        else:
            self.buffer = buffer

        self.reward_transformer = reward_transformer
        self.discount = discount
        self.policy_opt_steps = policy_opt_steps
        self.policy_lr = policy_lr

    def get_rewards(self, rewards):
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

    def select_action(self, timestep: dm_env.TimeStep) -> Union[int, np.array]:
        """
        Selects actions using current policy in an on-policy setting.

        @param timestep: the current timestep containg the observation
        @return: An action confirming to the dm_control numpy or int convention
        """
        obs_feat = torch.tensor([timestep.observation]).float()
        action = self.policy.sample(obs_feat)
        return action

    def report_tensorboard(
        self,
        observations,
        next_observations,
        rewards,
        actions,
        dist_before,
        dist_after,
        iteration,
    ):
        pol_loss = self.nll_loss(observations, next_observations, rewards, actions)
        kl_loss = torch.distributions.kl_divergence(dist_before, dist_after).mean(0)
        entropy = self.policy.distribution(observations).entropy().mean(0)

        logger.info(f"Iteration {iteration} done, " f"Reward {torch.sum(rewards)}")

        if self.writer is not None:
            self.writer.add_scalar("train/pol_loss", pol_loss, iteration)
            self.writer.add_scalar("train/reward", torch.sum(rewards), iteration)
            self.writer.add_histogram("train/actions", actions, iteration)
            self.writer.add_scalar("train/kl_loss_mean", kl_loss, iteration)
            self.writer.add_scalar("train/entropy_mean", entropy, iteration)

    def update(
        self, timestep: dm_env.TimeStep, action, new_timestep: dm_env.TimeStep,
    ):
        """Sampling: Obtain N samples (s_i, a_i, s_i', r_i)

        Currently done using the Agent interface of DM.
        """
        self.buffer.push(timestep, action, new_timestep)

    def optimize_loss(
        self,
        loss_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ],
        optimizer: torch.optim.Optimizer,
        optimizer_steps=300,
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
