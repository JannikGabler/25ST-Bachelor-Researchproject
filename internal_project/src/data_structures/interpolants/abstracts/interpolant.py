from abc import ABC, abstractmethod


class Interpolant(ABC):

    @abstractmethod
    def evaluate(self, x):
        pass
