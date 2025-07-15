import math
from fractions import Fraction

from exceptions.not_instantiable_error import NotInstantiableError


class NodesTestUtils:

    def __init__(self):
        raise NotInstantiableError(f"The class {repr(self.__class__.__name__)} can not be instantiated.")



    @staticmethod
    def chebyshev_2_nodes(interval: tuple[Fraction, Fraction], node_count: int) -> list[Fraction]:
        fractions: list[Fraction] = [Fraction(i / (node_count - 1)) for i in range(node_count)]
        values: list[float] = [math.cos(float(frac) * math.pi) for frac in fractions]
        non_transformed_nodes = [Fraction.from_float(value) for value in values]
        length = (interval[1] - interval[0]) / 2
        mid = (interval[0] + interval[1]) / 2
        return [length * frac + mid for frac in non_transformed_nodes]

