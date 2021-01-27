import torch.nn as nn


class SimpleValueFunction(nn.Module):
    def __init__(self, feature_shape):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(feature_shape, 1, bias=False))

    def forward(self, x):
        return self.model(x).squeeze()
