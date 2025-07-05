from data_structures.interpolants.abstracts.interpolant import Interpolant
import jax
import jax.numpy as jnp


class NewtonInterpolant(Interpolant):
    def __init__(self, nodes: jnp.ndarray, values: jnp.ndarray, weights: jnp.ndarray):
        self.nodes = nodes
        self.values = values
        self.weights = weights

    def evaluate(self, x:jnp.ndarray) -> jnp.ndarray:
        # Ensure input x is at least 1D to allow vectorized evaluation
        evaluation_points = jnp.atleast_1d(x)
        n = self.weights.size

        # Inner loop function implementing the nested form of the Newton polynomial, which corresponds to Horner's scheme for Newton form
        def horner_step(i, val):
            reverse_index = n - 1 - i
            return val * (evaluation_points - self.nodes[reverse_index]) + self.weights[reverse_index]

        # Initialize the array to hold the evaluated polynomial values
        polynomial_values = jnp.zeros_like(evaluation_points)

        # Evaluate the polynomial using Horner's method
        polynomial_values = jax.lax.fori_loop(0, n, horner_step, polynomial_values)

        # Return scalar if input was scalar, otherwise return array
        return polynomial_values if evaluation_points.ndim > 0 else polynomial_values.item()