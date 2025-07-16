from interpolants.abstracts.interpolant import Interpolant
import jax
import jax.numpy as jnp


class NewtonInterpolant(Interpolant):
    _divided_differences_: jnp.ndarray
    _values_: jnp.ndarray
    _nodes_: jnp.ndarray


    def __init__(self, nodes: jnp.ndarray, values: jnp.ndarray, divided_differences: jnp.ndarray):
        self._nodes_ = nodes
        self._values_ = values
        self._divided_differences_ = divided_differences


    def evaluate(self, x:jnp.ndarray) -> jnp.ndarray:
        # Ensure input x is at least 1D to allow vectorized evaluation
        evaluation_points = jnp.atleast_1d(x)
        n = self._divided_differences_.size

        # Inner loop function implementing the nested form of the Newton polynomial, which corresponds to Horner's scheme for Newton form
        def horner_step(i, val):
            reverse_index = n - 1 - i
            return val * (evaluation_points - self._nodes_[reverse_index]) + self._divided_differences_[reverse_index]

        # Initialize the array to hold the evaluated polynomial values
        polynomial_values = jnp.zeros_like(evaluation_points)

        # Evaluate the polynomial using Horner's method
        polynomial_values = jax.lax.fori_loop(0, n, horner_step, polynomial_values)

        # Return scalar if input was scalar, otherwise return array
        return polynomial_values if evaluation_points.ndim > 0 else polynomial_values.item()



    def __repr__(self) -> str:
        return f"NewtonInterpolant(weights={self._divided_differences_}, values={self._values_}, nodes={self._nodes_})"



    def __str__(self) -> str:
        return self.__repr__()