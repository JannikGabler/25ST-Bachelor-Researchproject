from fractions import Fraction

import jax.numpy as jnp

from exceptions.not_instantiable_error import NotInstantiableError


class EvaluationTestUtils:

    def __init__(self):
        raise NotInstantiableError(
            f"The class {repr(self.__class__.__name__)} can not be instantiated."
        )

    @staticmethod
    def evaluate_newton_polynom(
        divided_differences: list[Fraction], nodes: list[Fraction], x: list[Fraction]
    ) -> list[Fraction]:
        result: list[Fraction] = [Fraction(0) for _ in range(len(x))]

        for i in range(len(nodes) - 1, -1, -1):
            for j in range(len(x)):
                result[j] *= x[j] - nodes[i]
                result[j] += divided_differences[i]

        return result
