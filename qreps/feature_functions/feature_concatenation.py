import torch

from qreps.feature_functions.abstract_feature_function import (
    AbstractFeatureFunction,
    AbstractStateActionFeatureFunction,
)


class FeatureConcatenation(AbstractStateActionFeatureFunction):
    def __init__(
        self,
        obs_feature_fn: AbstractFeatureFunction,
        act_feature_fn: AbstractFeatureFunction,
    ):
        self.obs_feature_fn = obs_feature_fn
        self.act_feature_fn = act_feature_fn

    def __call__(self, state, action):
        super(FeatureConcatenation, self).__call__(state, action)
        obs_features = self.obs_feature_fn(state)
        act_features = self.act_feature_fn(action)
        if obs_features.ndim == 0 and act_features.ndim == 0:
            return torch.stack((obs_features, act_features))

        if act_features.ndim == 0:
            act_features = act_features.unsqueeze(0)

        if act_features.shape[0] != obs_features.shape[0]:
            # First dimension is not batch_dim
            act_features = act_features.unsqueeze(0)
            obs_features = obs_features.unsqueeze(0)

        if act_features.ndim < obs_features.ndim:
            act_features = act_features.unsqueeze(1)

        if obs_features.ndim < act_features.ndim:
            obs_features = obs_features.unsqueeze(1)

        return torch.cat((obs_features, act_features), dim=-1)
