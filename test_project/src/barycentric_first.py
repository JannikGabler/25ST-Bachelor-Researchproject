import jax
import jax.numpy as jnp


@jax.jit
def compute_weights(interpolation_nodes: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the barycentric weights for the first form of the barycentric interpolation formula.

    Args:
        interpolation_nodes: 1D array containing the distinct interpolation nodes.

    Returns:
        1D array containing the barycentric weights.
    """

    # Ensure input is a JAX array
    interpolation_nodes = jnp.asarray(interpolation_nodes)

    # Create a square matrix where each entry [j, k] is the difference between node j and node k
    # Note that the diagonal entries are all zero, since each node is subtracted from itself
    pairwise_diff = interpolation_nodes[:, None] - interpolation_nodes[None, :]

    # Create a boolean matrix with False on the diagonal and True elsewhere
    # This is used to exclude self-differences (which are zero) from the product
    mask = ~jnp.eye(len(interpolation_nodes), dtype=bool)

    # Replace diagonal entries (which are zero) with 1.0 to avoid affecting the product
    # Then compute the product across each row (over all k ≠ j)
    product = jnp.prod(jnp.where(mask, pairwise_diff, 1.0), axis=1)

    # Return the inverse of each product to get the barycentric weights
    return 1.0 / product


@jax.jit
def interpolate_single(x: float, interpolation_nodes: jnp.ndarray, function_values: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """
    Helper function for barycentric_type1_interpolate:
    Evaluates the barycentric interpolation polynomial of the first form at a single point.

    Args:
        x: Scalar evaluation point.
        interpolation_nodes: 1D array containing the distinct interpolation nodes.
        function_values: 1D array containing the function values at the nodes.
        weights: 1D array containing the barycentric weights.

    Returns:
         Interpolated function value at the evaluation point x.
         If x coincides with one of the interpolation nodes, the corresponding function value is returned exactly.
    """

    # Compute array of differences (x - x_j)
    diffs = x - interpolation_nodes

    # Create Boolean array where each difference is compared to zero: True where x == x_j and False elsewhere
    bool_diffs = jnp.equal(diffs, 0.0)

    # Extract function value f_j at node where x == x_j (sum picks this single matching value)
    # If x does not equal any node, sum is zero and will be ignored later
    exact_value = jnp.sum(jnp.where(bool_diffs, function_values, 0.0))

    # Replace zeros (where x == x_j) with 1.0 to avoid division by zero
    updated_diffs = jnp.where(bool_diffs, 1.0, diffs)

    # Calculate the node polynomial l(x)
    node_polynomial = jnp.prod(diffs)

    # Calculate the final value with the first form of the barycentric interpolation formula (Equation (5.9))
    interpolated_value = node_polynomial * jnp.sum((weights / updated_diffs) * function_values)

    # Return exact function value if x matches a node otherwise return interpolated value
    return jnp.where(jnp.any(bool_diffs), exact_value, interpolated_value)


@jax.jit
def barycentric_type1_interpolate(evaluation_points: jnp.ndarray, interpolation_nodes: jnp.ndarray, function_values: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluates the barycentric interpolation polynomial of the first form at specified points.
    The main calculation is done in the helper method interpolate_single.

    Args:
        evaluation_points:  A scalar or 1D array of evaluation points.
        interpolation_nodes: 1D array containing the distinct interpolation nodes.
        function_values: 1D array containing the function values at the nodes.

    Returns:
        1D array of interpolated values at each evaluation point.
        If evaluation_points was a scalar, a 1-element array is returned.
    """

    # Convert any array‐like to JAX arrays
    interpolation_nodes = jnp.asarray(interpolation_nodes)
    function_values = jnp.asarray(function_values)

    # Ensure input x is at least 1D to allow vectorized evaluation
    evaluation_points = jnp.atleast_1d(evaluation_points)

    # Compute the weights
    weights = compute_weights(interpolation_nodes)

    # Compute interpolation value for each evaluation point individually but efficiently
    return jax.vmap(lambda x: interpolate_single(x, interpolation_nodes, function_values, weights))(evaluation_points)