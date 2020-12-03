import torch

from .feature_functions import bellman_error, feature_difference


def test_feature_difference():
    features = torch.tensor([1, 2])
    features_next = torch.tensor([2, 3])
    diff = torch.tensor([1, 1])
    assert torch.equal(feature_difference(features, features_next), diff)


def test_bellman_error():
    theta = torch.tensor([2, 2])
    features = torch.tensor([1, 2])
    features_next = torch.tensor([2, 3])
    reward = torch.tensor(3)

    # result = 3 +  [2, 3]^T * [2, 2] - [1, 2]^T * [2, 2] = 3 + (4 + 6) + 2 + 4 = 7
    result = torch.tensor(7)
    bell_error_result = bellman_error(theta, features, features_next, reward)
    assert torch.equal(bell_error_result, result)
