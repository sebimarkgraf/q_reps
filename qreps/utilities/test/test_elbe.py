import torch

from qreps.utilities.elbe import empirical_bellman_error


class TestEmpiricalBellmanError(object):
    def test_shape(self):
        batch_size = 10
        feature_dim = 4
        features = torch.rand((batch_size, feature_dim))
        features_next = torch.rand((batch_size, feature_dim))
        actions = torch.randint(3, size=(batch_size,))
        rewards = torch.rand((batch_size,))

        def q_func(_obs, _a):
            return torch.rand((batch_size,))

        def v_func(_obs):
            return torch.rand((batch_size,))

        result = empirical_bellman_error(
            features, features_next, actions, rewards, q_func, v_func, discount=1.0
        )

        assert result.shape == torch.Size((batch_size,))
