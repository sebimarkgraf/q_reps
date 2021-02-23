import math

import torch

from qreps.valuefunctions.q_function import AbstractQFunction
from qreps.valuefunctions.value_functions import AbstractValueFunction


def empirical_bellman_error(
    features,
    features_next,
    actions,
    rewards,
    q_func: AbstractQFunction,
    v_func: AbstractValueFunction,
    discount,
):
    v_features = v_func(features_next)
    q_features = q_func(features, actions)
    return rewards + discount * v_features - q_features


def empirical_logistic_bellman(
    eta, features, features_next, actions, rewards, q_func, v_func, discount
):
    N = features.shape[0]
    exponents = torch.logsumexp(
        eta
        * empirical_bellman_error(
            features, features_next, actions, rewards, q_func, v_func, discount
        )
        + math.log(1 / N),
        0,
    )
    return 1 / eta * exponents
