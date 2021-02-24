import torch

from qreps.algorithms.sampler.abstract_sampler import AbstractSampler


class ExponentitedGradientSampler(AbstractSampler):
    def __init__(self, beta=0.1, *args, **kwargs):
        super(ExponentitedGradientSampler, self).__init__(*args, **kwargs)
        self.beta = beta
        self.h = torch.ones((self.length,))
        self.z = torch.ones((self.length,))
        self.z /= torch.sum(self.z)

    def get_next_distribution(self, bellman_error):
        self.z = self.z * torch.exp(torch.clamp(self.beta * self.h, -10, 10))
        self.z = self.z / (torch.sum(self.z))
        self.h = bellman_error.squeeze() - torch.log(self.length * self.z) / self.eta

        return torch.distributions.Categorical(probs=self.z)

    def get_distribution(self):
        return torch.distributions.Categorical(probs=self.z)
