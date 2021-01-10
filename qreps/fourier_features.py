import math

import torch

from qreps.util import to_torch


class FourierFeatures(torch.nn.Module):
    def __init__(self, num_states: int, num_features: int, gauss_scale=1.0):
        super().__init__()
        self.layer = torch.nn.Linear(num_states, num_features, bias=True)
        self.layer.requires_grad_(False)
        torch.nn.init.normal_(self.layer.weight, 0, gauss_scale)
        torch.nn.init.uniform_(self.layer.bias, 0, 2 * math.pi)

    def __call__(self, x):
        return torch.cos(self.layer(to_torch(x)))
