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
    eta, gamma, features, features_next, actions, rewards, v_func, q_func
):
    exponents = eta * empirical_bellman_error(
        features, features_next, actions, rewards, q_func, v_func, gamma
    )
    # Logmeanexp in pytorch of the empirical logistic bellman. Equation (11) in paper
    value = 1 / eta * torch.log(
        torch.mean(torch.exp(exponents - torch.max(exponents)))
    ) + 1 / eta * torch.exp(torch.max(exponents))
    # + (1 - gamma) * v_func(initial_state)

    return value
