import torch
import torch.testing
from torch.distributions import Categorical, MultivariateNormal

from qreps.feature_functions.feature_concatenation import FeatureConcatenation
from qreps.feature_functions.one_hot import OneHotFeature
from qreps.util import integrate


class TestConcatenation(object):
    def test_samefunction_concatenation(self):
        def iden_feature(x):
            return x.float()

        concatenator = FeatureConcatenation(
            obs_feature_fn=iden_feature, act_feature_fn=iden_feature
        )

        x = torch.tensor(2)
        a = torch.tensor(0)
        result = concatenator(x, a)
        torch.testing.assert_allclose(result, torch.tensor([2.0, 0.0]))

        x = torch.tensor([0, 1, 2])
        a = torch.tensor(2)
        torch.testing.assert_allclose(
            concatenator(x, a), torch.tensor([[0.0, 1.0, 2.0, 2.0]])
        )

        x = torch.tensor([0, 1, 2])
        a = torch.tensor([0, 1, 2])
        torch.testing.assert_allclose(
            concatenator(x, a), torch.tensor([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        )

        x = torch.tensor([[0, 0], [1, 1], [2, 2]])
        a = torch.tensor([0, 1, 2])
        torch.testing.assert_allclose(
            concatenator(x, a),
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        )

    def test_onehotencoded_concatenation(self):
        def iden_feature(x):
            return x.float()

        concatenator = FeatureConcatenation(
            obs_feature_fn=iden_feature, act_feature_fn=OneHotFeature(num_classes=2)
        )

        x = torch.tensor([0, 1, 2])
        a = torch.tensor(0)
        result = concatenator(x, a)
        assert torch.equal(result, torch.tensor([[0.0, 1.0, 2.0, 1.0, 0.0]]))

        concatenator = FeatureConcatenation(
            obs_feature_fn=OneHotFeature(num_classes=3), act_feature_fn=iden_feature
        )
        x = torch.tensor([0, 1, 2])
        a = torch.tensor([0, 1, 2])
        result = concatenator(x, a)
        assert torch.equal(
            result,
            torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 2.0]]
            ),
        )
