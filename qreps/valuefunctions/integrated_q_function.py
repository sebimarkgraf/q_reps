import torch

from qreps.policies import StochasticPolicy
from qreps.utilities.util import integrate

from .q_function import AbstractQFunction
from .value_functions import AbstractValueFunction


class IntegratedQFunction(AbstractValueFunction):
    def __init__(
        self,
        policy: StochasticPolicy,
        q_func: AbstractQFunction,
        alpha=1.0,
        *args,
        **kwargs
    ):
        super().__init__(obs_dim=q_func.n_obs, *args, **kwargs)
        self.alpha = alpha
        self.policy = policy
        self.q_func = q_func

    def forward(self, obs):
        def q_for_obs(action):
            return self.alpha * self.q_func(obs, action)

        distribution = self.policy.distribution(obs)
        values = 1 / self.alpha * integrate(q_for_obs, distribution)
        return values.squeeze(-1)
