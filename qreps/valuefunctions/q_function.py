from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from qreps.feature_functions.abstract_feature_function import (
    AbstractStateActionFeatureFunction,
)


class AbstractQFunction(nn.Module, metaclass=ABCMeta):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.n_obs = obs_dim
        self.n_action = act_dim

    @abstractmethod
    def forward(self, observation, action):
        pass

    @abstractmethod
    def features(self, observation, action):
        pass


class SimpleQFunction(AbstractQFunction):
    def __init__(self, feature_fn: AbstractStateActionFeatureFunction, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_fn = feature_fn
        self.model = nn.Linear(self.n_obs + self.n_action, 1, bias=False)

    def features(self, observation, action):
        return self.feature_fn(observation, action)

    def forward(self, observation, action):
        input = self.features(observation, action)
        return self.model(input).squeeze(-1)


class NNQFunction(AbstractQFunction):
    def __init__(self, feature_fn: AbstractStateActionFeatureFunction, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_fn = feature_fn
        self.model = nn.Sequential(
            nn.Linear(self.n_obs + self.n_action, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def features(self, observation, action):
        return self.feature_fn(observation, action)

    def forward(self, observation, action):
        input = self.features(observation, action)
        return self.model(input).squeeze(-1)
