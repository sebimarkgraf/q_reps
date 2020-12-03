from torch import Tensor


def bellmann_error(theta, features: Tensor, features_next: Tensor, rewards: Tensor):
    """Calculates the bellmann TD error for the features"""
    return rewards + theta.dot(features_next) - theta.dot(features)


def feature_difference(features: Tensor, features_next: Tensor):
    """Calculates the feature difference between the features of s and s'
    @param features: torch tensor in the shape of samples, feature_dim
    @param features_next: torch in the shape of samples, feature_dim
    @return: the feature difference (s' - s)
    """
    return features_next - features
