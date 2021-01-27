import torch.nn as nn


class SimpleValueFunction(nn.Module):
    def __init__(self, feature_shape, feature_fn=nn.Identity()):
        super().__init__()
        self.feature_fn = feature_fn
        self.model = nn.Sequential(nn.Linear(feature_shape, 1, bias=False))

    def forward(self, x):
        return self.model(self.feature_fn(x)).squeeze()


class NNValueFunction(nn.Module):
    def __init__(self, feature_shape, feature_fn=nn.Identity()):
        super().__init__()
        self.feature_fn = feature_fn
        self.model = nn.Sequential(
            nn.Linear(feature_shape, 128), nn.ReLU(), nn.Linear(128, 1, bias=False)
        )

    def forward(self, x):
        return self.model(self.feature_fn(x)).squeeze()
