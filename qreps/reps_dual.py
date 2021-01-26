import torch
from torch import Tensor

from qreps.feature_functions import bellman_error_batched


def reps_dual(
    eta: Tensor,
    theta: Tensor,
    features: Tensor,
    features_next: Tensor,
    rewards: Tensor,
    epsilon: Tensor,
    gamma: float,
    l2_reg_dual: float = 0.0,
):
    """
    Implements REPS loss

    @param eta: Current value for parameter eta
    @param theta:  Current value for value function parameters
    @param features: the batched features of the state
    @param features_next: the batches features for the next state i.e. features[1:]
    @param rewards: the batches rewards for the transitions
    @param epsilon: KL_Loss divergence parameter
    @param l2_reg_dual: Parameter to regularize the size of eta
    @return: the calculated dual function value, supporting autograd of PyTorch
    """
    bellmann_error = bellman_error_batched(
        theta, features, features_next, rewards, gamma
    )

    # LogMeanExp manually, due to PyTorch only supporting logsumexp
    max_bell = torch.max(bellmann_error / eta)
    dual_val = (
        eta * epsilon
        + eta * torch.log(torch.mean(torch.exp(bellmann_error / eta - max_bell)))
        + eta * max_bell
    )

    # Allow Regularization to keep eta smaller
    dual_val += l2_reg_dual * (torch.square(eta) + torch.square(1 / eta))

    return dual_val
