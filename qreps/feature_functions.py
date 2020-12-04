from torch import Tensor


def bellman_error(
    theta: Tensor, features: Tensor, features_next: Tensor, reward: Tensor
) -> Tensor:
    """Calculates the bellman error for the features"""
    return reward + theta.dot(features_next) - theta.dot(features)


def bellman_error_batched(
    theta: Tensor, features: Tensor, features_next: Tensor, reward: Tensor
):
    return reward + theta.matmul(features_next.T) - theta.matmul(features.T)


def feature_difference(features: Tensor, features_next: Tensor) -> Tensor:
    """Calculates the feature difference between the features of s and s'
    @param features: torch tensor in the shape of samples, feature_dim
    @param features_next: torch in the shape of samples, feature_dim
    @return: the feature difference (s' - s)
    """
    return features_next - features
