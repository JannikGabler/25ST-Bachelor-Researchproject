from data_structures.interpolants.abstracts.interpolant import Interpolant
import jax
import jax.numpy as jnp


class BarycentricType2Interpolant(Interpolant):
    def __init__(self, nodes: jnp.ndarray, values: jnp.ndarray, weights: jnp.ndarray):
        self.nodes = nodes
        self.values = values
        self.weights = weights

    @jax.jit
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
        diffs = x - self.nodes

        # Check if x exactly matches any interpolation node (True where difference is zero)
        exact_match = jnp.isclose(diffs, 0.0)

        # Return the exact function value if x matches a node
        # Otherwise compute the barycentric interpolation using weights and differences
        return jnp.where(
            jnp.any(exact_match),
            self.values[jnp.argmax(exact_match)],
            jnp.sum((self.weights * self.values) / diffs) / jnp.sum(self.weights / diffs)
        )

    def evaluate(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self._interpolate_single)(x)

