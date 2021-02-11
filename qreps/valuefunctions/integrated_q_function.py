from qreps.policies.stochasticpolicy import StochasticPolicy
from qreps.util import integrate
from qreps.valuefunctions.q_function import AbstractQFunction
from qreps.valuefunctions.value_functions import AbstractValueFunction


class IntegratedQFunction(AbstractValueFunction):
    def __init__(
        self,
        policy: StochasticPolicy,
        q_func: AbstractQFunction,
        alpha=1.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.policy = policy
        self.q_func = q_func

    def forward(self, obs):
        def q_for_obs(action):
            return self.q_func(action)

        return integrate(q_for_obs, self.policy.distribution(obs))
