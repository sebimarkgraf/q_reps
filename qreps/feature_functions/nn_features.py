import torch.nn as nn

from .abstract_feature_function import AbstractFeatureFunction


class NNFeatures(AbstractFeatureFunction):
    def __init__(self, obs_dim, feat_dim=200):
        super(NNFeatures, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 200), nn.ReLU(), nn.Linear(200, feat_dim), nn.ReLU(),
        )
        # Let's freeze it
        self.model.requires_grad_(False)

    def __call__(self, state):
        return self.model.forward(state)
