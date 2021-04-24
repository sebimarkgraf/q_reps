import torch.nn as nn

from .abstract_feature_function import AbstractFeatureFunction


class CNNFeatures(AbstractFeatureFunction):
    def __init__(self, obs_dim, in_channels, feat_dim=200):
        super(CNNFeatures, self).__init__()
        n, m = obs_dim
        embedding = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(embedding, 200),
            nn.ReLU(),
            nn.Linear(200, feat_dim),
            nn.ReLU(),
        )
        # Let's freeze it
        self.model.requires_grad_(False)

    def __call__(self, state):
        return self.model.forward(state)
