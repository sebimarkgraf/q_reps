import pytest
import torch
import torch.testing

from qreps.feature_functions.feature_concatenation import FeatureConcatenation
from qreps.feature_functions.one_hot import OneHotFeature
from qreps.policies.categorical_mlp import CategoricalMLP
from qreps.policies.stochastic_table import StochasticTablePolicy
from qreps.valuefunctions.integrated_q_function import IntegratedQFunction
from qreps.valuefunctions.q_function import SimpleQFunction


@pytest.fixture(params=[1, 2])
def act_dim(request):
    return request.param


@pytest.fixture(params=[4])
def obs_dim(request):
    return request.param


@pytest.fixture(params=[None, 10])
def batch_size(request):
    return request.param


class TestIntegratedQFunction(object):
    def init(
        self, obs_dim, act_dim,
    ):
        def feature(x):
            return x.float()

        feature_fn = FeatureConcatenation(
            obs_feature_fn=feature, act_feature_fn=OneHotFeature(num_classes=act_dim)
        )

        self.q_function = SimpleQFunction(
            obs_dim=obs_dim, act_dim=act_dim, feature_fn=feature_fn
        )
        self.policy = CategoricalMLP(obs_shape=obs_dim, act_shape=act_dim)
        self.value_function = IntegratedQFunction(
            q_func=self.q_function, policy=self.policy
        )

    def test_creation(self, obs_dim, act_dim, batch_size):
        self.init(obs_dim, act_dim)

    def test_forward(self, obs_dim, act_dim, batch_size):
        self.init(obs_dim, act_dim)

        if batch_size is not None:
            state = torch.rand(size=(batch_size, obs_dim))
        else:
            state = torch.rand(size=(obs_dim,))

        value = self.value_function(state)

        if batch_size is not None:
            assert value.shape == torch.Size([batch_size])
        else:
            assert value.ndim == 0
