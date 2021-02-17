import pytest
import torch
import torch.testing
from torch.distributions import Categorical, MultivariateNormal

from qreps.util import integrate


class TestIntegrate(object):
    def test_discrete_distribution(self):
        d = Categorical(torch.tensor([0.1, 0.2, 0.3, 0.4]))

        def _function(a):
            return 2 * a

        torch.testing.assert_allclose(integrate(_function, d), 4.0)

    def test_multivariate_normal(self):
        d = MultivariateNormal(torch.tensor([0.2]), scale_tril=1e-6 * torch.eye(1))

        def _function(a):
            return 2 * a

        torch.testing.assert_allclose(
            integrate(_function, d, continuous_samples=100), 0.4, rtol=1e-3, atol=1e-3
        )
