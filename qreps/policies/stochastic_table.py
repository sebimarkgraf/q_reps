import torch
import torch.nn as nn

from qreps.policies.policy import Policy


class StochasticTablePolicy(Policy, nn.Module):
    def __init__(self, n_states: int, n_actions: int, *args, **kwargs):
        super(StochasticTablePolicy, self).__init__(*args, **kwargs)

        # Initialize with same prob for all actions in each state
        _policy = torch.ones((n_states, n_actions))
        _policy /= torch.sum(_policy, 1, keepdim=True)
        self._policy = nn.Parameter(_policy)

    def forward(self, x):
        return super(StochasticTablePolicy, self).forward(x)

    def _dist(self, observation):
        return torch.distributions.Categorical(
            logits=self._policy[self.forward(observation)]
        )

    def log_likelihood(self, feat, taken_actions) -> torch.FloatTensor:
        return self._dist(feat).log_prob(taken_actions)

    @torch.no_grad()
    def sample(self, observation):
        if self._stochastic:
            return self._dist(observation).sample().item()
        else:
            return torch.argmax(self._policy[self.forward(observation)]).item()
