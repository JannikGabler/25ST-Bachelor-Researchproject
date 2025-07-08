from data_structures.interpolants.abstracts.interpolant import Interpolant
import jax
import jax.numpy as jnp


class BarycentricSecondInterpolant(Interpolant):
    _nodes_: jnp.ndarray
    _values_: jnp.ndarray
    _weights_: jnp.ndarray



    def __init__(self, nodes: jnp.ndarray, values: jnp.ndarray, weights: jnp.ndarray):
        self._nodes_ = nodes
        self._values_ = values
        self._weights_ = weights



    def _interpolate_single(self, x: float) -> jnp.ndarray:
        """
        Helper function for barycentric_type2_interpolate:
        Evaluates the barycentric interpolation polynomial of the second form at a single point.

        Args:
            x: Scalar evaluation point.

        Returns:
             Interpolated function value at the evaluation point x.
             If x coincides with one of the interpolation nodes, the corresponding function value is returned exactly.
        """

        # Compute array of differences (x - x_j)
        diffs = x - self._nodes_

        # Check if x exactly matches any interpolation node (True where difference is zero)
        exact_match = jnp.isclose(diffs, 0.0)

        # Return the exact function value if x matches a node
        # Otherwise compute the barycentric interpolation using weights and differences
        return jnp.where(
            jnp.any(exact_match),
            self._values_[jnp.argmax(exact_match)],
            jnp.sum((self._weights_ * self._values_) / diffs) / jnp.sum(self._weights_ / diffs)
        )



    def evaluate(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.jit(jax.vmap(lambda x_i: self._interpolate_single(x_i)))(x)



    def __repr__(self) -> str:
        return f"BarycentricSecondInterpolant(weights={self._weights_}, values={self._values_}, nodes={self._nodes_})"



    def __str__(self) -> str:
        return self.__repr__()



    def __eq__(self, other):
        if not isinstance(other, BarycentricSecondInterpolant):
            return False
        else:
            return jnp.array_equal(self._weights_, other._weights_) and jnp.array_equal(self._nodes_, other._nodes_)
