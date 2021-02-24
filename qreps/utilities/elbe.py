from qreps.utilities.math import logmeanexp
from qreps.valuefunctions import AbstractQFunction, AbstractValueFunction


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
    errors = empirical_bellman_error(
        features, features_next, actions, rewards, q_func, v_func, discount
    )
    return 1 / eta * logmeanexp(eta * errors, dim=0)
