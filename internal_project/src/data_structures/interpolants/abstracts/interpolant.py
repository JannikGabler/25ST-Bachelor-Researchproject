from abc import ABC, abstractmethod


class Interpolant(ABC):

    @abstractmethod
    def evaluate(self, x):
        pass

    @abstractmethod
    def vectorized_evaluate(self, x):
        pass