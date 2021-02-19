import torch
import torch.nn.functional as F

from qreps.algorithms.sampler.abstract_sampler import AbstractSampler


class ExponentitedGradientSampler(AbstractSampler):
    def __init__(self, beta=0.1, *args, **kwargs):
        super(ExponentitedGradientSampler, self).__init__(*args, **kwargs)
        self.beta = beta
        self.h = torch.ones((self.length,))
        self.z = torch.ones((self.length,))
        self.z /= torch.sum(self.z)

    def get_next_distribution(self, bellman_error):
        self.z = self.z * torch.exp(self.beta * self.h)
        self.z = F.softmax(self.z, dim=0)
        self.h = bellman_error.squeeze() - torch.log(self.length * self.z) / self.eta

        return self.z
