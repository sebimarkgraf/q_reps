from abc import ABCMeta, abstractmethod


class AbstractSampler(metaclass=ABCMeta):
    def __init__(self, length, eta):
        self.length = length
        self.eta = eta

    @abstractmethod
    def get_next_distribution(self, bellman_error):
        pass
