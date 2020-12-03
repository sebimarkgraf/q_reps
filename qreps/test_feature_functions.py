import torch

from .feature_functions import feature_difference


def test_feature_difference():
    features = torch.tensor([1, 2])
    features_next = torch.tensor([2, 3])
    diff = torch.tensor([1, 1])
    assert torch.equal(feature_difference(features, features_next), diff)
