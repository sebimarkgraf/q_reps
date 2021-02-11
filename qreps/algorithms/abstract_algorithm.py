from abc import ABCMeta

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from qreps.memory.replay_buffer import ReplayBuffer

DEFAULT_REPLAY_BUFFER_SIZE = 100000


class AbstractAlgorithm(nn.Module, metaclass=ABCMeta):
    def __init__(self, writer: SummaryWriter = None, buffer=None):
        super().__init__()
        self.writer = writer
        if buffer is None:
            self.buffer = ReplayBuffer(DEFAULT_REPLAY_BUFFER_SIZE)
        else:
            self.buffer = buffer

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

        if self.writer is not None:
            self.writer.add_scalar("train/pol_loss", pol_loss, iteration)
            self.writer.add_scalar("train/reward", torch.sum(rewards), iteration)
            self.writer.add_histogram("train/actions", actions, iteration)
            self.writer.add_scalar("train/kl_loss_mean", kl_loss, iteration)
            self.writer.add_scalar("train/entropy_mean", entropy, iteration)
