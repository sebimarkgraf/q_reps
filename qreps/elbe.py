import torch


def empirical_bellman_error(
    features, features_next, actions, rewards, q_func, v_func, discount
):
    return rewards + discount * v_func(features_next) - q_func(features, actions)


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
