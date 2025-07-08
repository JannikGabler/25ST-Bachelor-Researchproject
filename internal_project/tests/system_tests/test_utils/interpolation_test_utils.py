from fractions import Fraction

import jax.numpy as jnp

from exceptions.not_instantiable_error import NotInstantiableError


class InterpolationTestUtils:


    def __init__(self):
        raise NotInstantiableError(f"The class {repr(self.__class__.__name__)} can not be instantiated.")



    @staticmethod
    def calc_divided_differences(nodes: list[Fraction], values: jnp.ndarray) -> list[Fraction]:
        size = len(nodes)

        divided_differences: list[list[Fraction]] = [[Fraction(0) for _ in range(size)] for _ in range(size)]

        for i in range(size):
            divided_differences[0][i] = Fraction.from_float(values[i].astype(float).item())

        for k in range(1, size):
            for i in range(k, size):
                new_value: Fraction = (divided_differences[k-1][i] - divided_differences[k-1][i-1]) / (nodes[i] - nodes[i-k])
                divided_differences[k][i] = new_value

        return [divided_differences[i][i] for i in range(size)]







