import torch
import torch.nn as nn

from qreps.policies.policy import Policy


class GaussianMLPPolicy(Policy, nn.Module):
    """Gaussian Multi Layer Perceptron as a Policy.

    Estimates mean of a gaussian distribution for every action and the corresponding deviation
    depending on the given observations.

    When set to eval mode returns the mean as action for every observation.
    """

    def __init__(
        self, obs_shape, act_shape, action_min, action_max, sigma=1.0, *args, **kwargs
    ):
        super(GaussianMLPPolicy, self).__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Linear(obs_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_shape),
        )
        self.log_sigma = nn.Parameter(torch.tensor(sigma))
        self.action_max = action_max
        self.action_min = action_min

    def forward(self, x):
        return self.model(super(GaussianMLPPolicy, self).forward(x))

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    def _dist(self, observation) -> torch.distributions.Distribution:
        return torch.distributions.normal.Normal(self.forward(observation), self.sigma)

    @torch.no_grad()
    def sample(self, observation):
        if self._stochastic:
            return torch.clamp(
                self._dist(observation).sample(), self.action_min, self.action_max
            )
        else:
            return torch.clamp(
                self.forward(observation), self.action_min, self.action_max
            )

    def log_likelihood(self, feat, taken_actions) -> torch.FloatTensor:
        return self._dist(feat).log_prob(taken_actions)
