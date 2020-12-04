import torch
from torch import Tensor

from qreps.feature_functions import bellman_error_batched, feature_difference


def reps_dual(
    eta: Tensor,
    theta: Tensor,
    features: Tensor,
    features_next: Tensor,
    rewards: Tensor,
    epsilon: Tensor,
):
    bellmann_error = bellman_error_batched(theta, features, features_next, rewards)
    feat_diff = feature_difference(features, features_next)

    dual_val = eta * epsilon + eta * (
        torch.logsumexp(bellmann_error / eta, 0)
        - torch.log(torch.tensor(features.shape[0], dtype=torch.get_default_dtype()))
    )

    bellman_sum = torch.sum(bellmann_error / eta)
    d_eta = (
        epsilon
        + torch.logsumexp(bellmann_error / eta, 0)
        - torch.sum(
            torch.exp(bellmann_error / eta - bellman_sum) / eta * bellmann_error
        )
    )
    d_theta = (torch.exp(bellmann_error / eta - bellman_sum)).matmul(feat_diff)

    return dual_val, d_eta, d_theta
