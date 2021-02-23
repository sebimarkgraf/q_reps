import torch

from qreps.algorithms.sampler.abstract_sampler import AbstractSampler


class BestResponseSampler(AbstractSampler):
    """
    Best Response Sampler that directly gives a distribution according to the bellman errors.
    Useful for problems where the exploration space is huge.

    Used in Logistic Q-Learning for CartPole.
    """

    def __init__(self, *args, **kwargs):
        super(BestResponseSampler, self).__init__(*args, **kwargs)
        self.log_dist = torch.zeros((self.length,))

    def get_next_distribution(self, bellman):
        self.log_dist = self.eta * bellman
        return torch.distributions.Categorical(logits=self.log_dist)

    def get_distribution(self):
        return torch.distributions.Categorical(logits=self.log_dist)
