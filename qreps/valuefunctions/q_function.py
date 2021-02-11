from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn


class AbstractQFunction(nn.Module, metaclass=ABCMeta):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.n_obs = obs_dim
        self.n_action = act_dim

    @abstractmethod
    def forward(self, observation, action):
        pass


class NNQFunction(AbstractQFunction):
    def __init__(self, feature_fn, act_feature_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_fn = feature_fn
        self.act_fn = act_feature_fn
        self.model = nn.Sequential(
            nn.Linear(self.n_obs + self.n_action, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, observation, action):
        input = torch.cat([self.feature_fn(observation), self.act_fn(action)], -1)
        return self.model(input).squeeze()
