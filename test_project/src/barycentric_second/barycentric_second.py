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
    bool_diff = ~jnp.eye(len(interpolation_nodes), dtype=bool)

    # Replace diagonal entries (which are zero) with 1.0 to avoid affecting the product
    # Then compute the product across each row (over all k ≠ j)
    product = jnp.prod(jnp.where(bool_diff, pairwise_diff, 1.0), axis=1)

    # Divide 1.0 by the product to get the barycentric weights (Equation (5.6))
    return 1.0 / product

