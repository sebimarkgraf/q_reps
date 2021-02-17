import torch.nn.functional as F

from qreps.feature_functions.abstract_feature_function import AbstractFeatureFunction


class OneHotFeature(AbstractFeatureFunction):
    def __init__(self, num_classes: int):
        super(OneHotFeature, self).__init__()
        self.num_classes = num_classes

    def __call__(self, value):
        super(OneHotFeature, self).__call__(value)
        return F.one_hot(value.long(), num_classes=self.num_classes).float()
