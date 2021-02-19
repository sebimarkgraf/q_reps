import torch.nn as nn

from qreps.feature_functions.abstract_feature_function import (
    AbstractStateActionFeatureFunction,
)
from qreps.feature_functions.feature_concatenation import FeatureConcatenation
from qreps.feature_functions.identity import IdentityFeature
from qreps.feature_functions.one_hot import OneHotFeature


class NNFeatures(AbstractStateActionFeatureFunction):
    def __init__(self, action_num, obs_dim):
        super(NNFeatures, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim + action_num, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
        )
        # Let's freeze it
        self.model.requires_grad_(False)
        self.concat = FeatureConcatenation(
            obs_feature_fn=IdentityFeature(), act_feature_fn=OneHotFeature(action_num)
        )

    def __call__(self, state, action):
        return self.model.forward(self.concat(state, action))
