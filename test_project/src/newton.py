import jax
import jax.numpy as jnp


@jax.jit
def divided_differences(interpolation_nodes: jnp.ndarray, function_values: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the coefficients of the Newton interpolation polynomial using divided differences.
    These coefficients can then be used to evaluate the interpolation polynomial efficiently in Newton form.

    Args:
         interpolation_nodes: 1D array of distinct interpolation nodes.
         function_values: 1D array of function values f(x) corresponding to the nodes.

    Returns:
         coef: 1D array of divided differences.
    """

    # Number of interpolation points
    n = interpolation_nodes.size

    # Initialize coefficients array with function values
    coefficients = function_values.copy()

    # Compute divided differences of increasing order
    def outer_loop(j, coef):
        # Update entries for current order of divided differences
        def inner_loop(i, coef_inner):
            numerator = coef[i] - coef[i - 1]
            denominator = interpolation_nodes[i] - interpolation_nodes[i - j]
            return coef_inner.at[i].set(numerator/denominator)

        # Apply the inner loop to update coefficients for the current order
        coef = jax.lax.fori_loop(j, n, inner_loop, coef)
        return coef

    # Apply the outer loop to compute all orders of divided differences
    coefficients = jax.lax.fori_loop(1, n, outer_loop, coefficients)

    return coefficients


@jax.jit
def newton_interpolate(evaluation_points: jnp.ndarray, interpolation_nodes: jnp.ndarray, coefficents: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluates the Newton interpolation polynomial at given points.
    The polynomial is defined by the coefficients and the interpolation nodes.

    Args:
         evaluation_points: Scalar or 1D array of evaluation points.
         interpolation_nodes: 1D array of the original interpolation nodes.
         coefficents: 1D array of divided difference coefficients computed via 'divided_differences'.

    Returns:
         polynomial_values: 1D array containing the evaluated polynomial values p(x) or scalar if input was scalar.
    """

    # Ensure input x is at least 1D to allow vectorized evaluation
    evaluation_points = jnp.atleast_1d(evaluation_points)
    n = coefficents.size

    # Inner loop function implementing the nested form of the Newton polynomial, which corresponds to Horner's scheme for Newton form
    def horner_step(i, val):
        reverse_index = n - 1 - i
        return val * (evaluation_points - interpolation_nodes[reverse_index]) + coefficents[reverse_index]

    # Initialize the array to hold the evaluated polynomial values
    polynomial_values = jnp.zeros_like(evaluation_points)

    # Evaluate the polynomial using Horner's method
    polynomial_values = jax.lax.fori_loop(0, n, horner_step, polynomial_values)

    # Return scalar if input was scalar, otherwise return array
    return polynomial_values if evaluation_points.ndim > 0 else polynomial_values.item()
