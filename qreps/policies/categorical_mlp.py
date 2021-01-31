import torch
import torch.nn as nn

from qreps.policies.stochasticpolicy import StochasticPolicy


class CategoricalMLP(StochasticPolicy, nn.Module):
    def __init__(self, obs_shape, act_shape, *args, **kwargs):
        super(CategoricalMLP, self).__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Linear(obs_shape, 128),
            nn.ReLU(),
            nn.Linear(128, act_shape),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.model(super(CategoricalMLP, self).forward(x))

    def distribution(self, observation) -> torch.distributions.Distribution:
        return torch.distributions.categorical.Categorical(self.forward(observation))

    @torch.no_grad()
    def sample(self, observation):
        if self._stochastic:
            return self.distribution(observation).sample().item()
        else:
            return torch.argmax(self.forward(observation)).item()

    def log_likelihood(self, features, actions):
        return self.distribution(features).log_prob(actions)
