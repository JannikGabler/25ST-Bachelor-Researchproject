from data_structures.interpolants.abstracts.interpolant import Interpolant
import jax
import jax.numpy as jnp



class BarycentricFirstInterpolant(Interpolant):
    _weights_: jnp.ndarray
    _values_: jnp.ndarray
    _nodes_: jnp.ndarray



    def __init__(self, nodes: jnp.ndarray, values: jnp.ndarray, weights: jnp.ndarray):
        self._nodes_ = nodes
        self._values_ = values
        self._weights_ = weights



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
        diffs = x - self._nodes_

        # Create Boolean array where each difference is compared to zero: True where x == x_j and False elsewhere
        bool_diffs = jnp.equal(diffs, 0.0)

        # Extract function value f_j at node where x == x_j (sum picks this single matching value)
        # If x does not equal any node, sum is zero and will be ignored later
        exact_value = jnp.sum(jnp.where(bool_diffs, self._values_, 0.0))

        # Replace zeros (where x == x_j) with 1.0 to avoid division by zero
        updated_diffs = jnp.where(bool_diffs, 1.0, diffs)

        # Calculate the node polynomial l(x)
        node_polynomial = jnp.prod(diffs)

        # Calculate the final value with the first form of the barycentric interpolation formula (Equation (5.9))
        interpolated_value = node_polynomial * jnp.sum((self._weights_ / updated_diffs) * self._values_)

        # Return exact function value if x matches a node otherwise return interpolated value
        return jnp.where(jnp.any(bool_diffs), exact_value, interpolated_value)



    def evaluate(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.jit(jax.vmap(lambda x_i: self._interpolate_single(x_i)))(x)



    def __repr__(self) -> str:
        return f"BarycentricFirstInterpolant(weights={self._weights_}, values={self._values_}, nodes={self._nodes_})"



    def __str__(self) -> str:
        return self.__repr__()



    def __eq__(self, other):
        if not isinstance(other, BarycentricFirstInterpolant):
            return False
        else:
            return jnp.array_equal(self._weights_, other._weights_) and jnp.array_equal(self._nodes_, other._nodes_)

