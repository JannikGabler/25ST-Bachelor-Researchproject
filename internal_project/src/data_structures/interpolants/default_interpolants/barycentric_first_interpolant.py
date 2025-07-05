from data_structures.interpolants.abstracts.interpolant import Interpolant
import jax
import jax.numpy as jnp


class BarycentricType1Interpolant(Interpolant):
    def __init__(self, nodes: jnp.ndarray, values: jnp.ndarray, weights: jnp.ndarray):
        self.nodes = nodes
        self.values = values
        self.weights = weights

    @jax.jit
    def _interpolate_single(self, x: float) -> jnp.ndarray:
        """
            Helper function for barycentric_type1_interpolate:
            Evaluates the barycentric interpolation polynomial of the first form at a single point.

            Args:
                x: Scalar evaluation point.

            Returns:
                 Interpolated function value at the evaluation point x.
                 If x coincides with one of the interpolation nodes, the corresponding function value is returned exactly.
        """

        # Compute array of differences (x - x_j)
        diffs = x - self.nodes

        # Create Boolean array where each difference is compared to zero: True where x == x_j and False elsewhere
        bool_diffs = jnp.equal(diffs, 0.0)

        # Extract function value f_j at node where x == x_j (sum picks this single matching value)
        # If x does not equal any node, sum is zero and will be ignored later
        exact_value = jnp.sum(jnp.where(bool_diffs, self.values, 0.0))

        # Replace zeros (where x == x_j) with 1.0 to avoid division by zero
        updated_diffs = jnp.where(bool_diffs, 1.0, diffs)

        # Calculate the node polynomial l(x)
        node_polynomial = jnp.prod(diffs)

        # Calculate the final value with the first form of the barycentric interpolation formula (Equation (5.9))
        interpolated_value = node_polynomial * jnp.sum((self.weights / updated_diffs) * self.values)

        # Return exact function value if x matches a node otherwise return interpolated value
        return jnp.where(jnp.any(bool_diffs), exact_value, interpolated_value)


    def evaluate(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self._interpolate_single)(x)
