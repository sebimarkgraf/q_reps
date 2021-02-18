import torch

from qreps.feature_functions.one_hot import OneHotFeature


class TestOneHot(object):
    def test_onehot_float(self):
        onehot = OneHotFeature(num_classes=3)

        x = torch.tensor(1.0)
        result = onehot(x)
        torch.testing.assert_allclose(result, [0.0, 1.0, 0.0])

        x = torch.tensor([0.0, 1.0, 2.0])
        result = onehot(x)
        torch.testing.assert_allclose(
            result, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        x = torch.tensor([[0.0], [1.0], [2.0]])
        result = onehot(x)
        torch.testing.assert_allclose(
            result, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        x = torch.tensor([[[1.0], [2.0]]])
        result = onehot(x)
        torch.testing.assert_allclose(result, [[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])

    def test_onehot_long(self):
        onehot = OneHotFeature(num_classes=2)

        x = torch.tensor([0, 1], dtype=torch.long)
        result = onehot(x)
        torch.testing.assert_allclose(result, [[1.0, 0.0], [0.0, 1.0]])
