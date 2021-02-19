import torch

from qreps.algorithms.sampler.abstract_sampler import AbstractSampler


class BestResponseSampler(AbstractSampler):
    """
    Best Response Sampler that directly gives a distribution according to the bellman errors.
    Useful for problems where the exploration space is huge.

    Used in Logistic Q-Learning for CartPole.
    """

    def get_next_distribution(self, bellman):
        return torch.exp(self.eta * bellman)
