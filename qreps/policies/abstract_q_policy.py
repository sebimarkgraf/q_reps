"""Interface for Q-Function derived policies."""

from abc import ABCMeta

from .stochasticpolicy import StochasticPolicy


class AbstractQFunctionPolicy(StochasticPolicy, metaclass=ABCMeta):
    def __init__(self, q_function, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_function = q_function
