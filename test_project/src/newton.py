import jax
import jax.numpy as jnp

def divided_differences(interpolation_nodes: jnp.ndarray, function_values: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the coefficients of the Newton interpolation polynomial using divided differences.
    These coefficients can then be used to evaluate the interpolation polynomial efficiently in Newton form.

    Parameters
    ----------
    interpolation_nodes: 1D array of interpolation nodes, must be distinct.
    function_values: 1D array of function values f(x) corresponding to the nodes.

    Returns
    -------
    coef: 1D array of divided differences a_k, starting with a_0 = f[x_0], a_1 = f[x_0, x_1], etc.
    """

    if interpolation_nodes.size == 0:
        raise ValueError("Interpolation nodes must not be empty.")
    if function_values.size == 0:
        raise ValueError("Function values must not be empty.")
    if interpolation_nodes.size != function_values.size:
        raise ValueError("Interpolation nodes and function values must have the same length.")
    if jnp.unique(interpolation_nodes).size != interpolation_nodes.size:
        raise ValueError("Interpolation nodes must be distinct.")

    n = interpolation_nodes.size
    coef = function_values.copy()

    for j in range(1, n):
        numerator = coef[j:n] - coef[j - 1: n - 1]
        denominator = interpolation_nodes[j:n] - interpolation_nodes[0 : n - j]
        coef = coef.at[j:n].set(numerator / denominator)

    return coef


def newton_interpolate(evaluation_points: jnp.ndarray, interpolation_nodes: jnp.ndarray, coef: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluates the Newton interpolation polynomial at given points.
    The polynomial is defined by the coefficients and the interpolation nodes.

    Parameters
    ----------
    evaluation_points: Scalar or 1D array of evaluation points.
    interpolation_nodes: 1D array of the original interpolation nodes.
    coef: 1D array of divided difference coefficients computed via 'divided_differences'.

    Returns
    -------
    polynomial_values: Array containing the evaluated polynomial values p(x).
    """

    if interpolation_nodes.size == 0:
        raise ValueError("Interpolation nodes must not be empty.")
    if coef.size == 0:
        raise ValueError("Divided difference coefficients must not be empty.")
    if interpolation_nodes.size != coef.size:
        raise ValueError("Interpolation nodes and coefficients must have the same length.")
    if jnp.unique(interpolation_nodes).size != interpolation_nodes.size:
        raise ValueError("Interpolation nodes must be distinct.")

    # Ensure input x is at least 1D to allow vectorized evaluation
    evaluation_points = jnp.atleast_1d(evaluation_points)
    n = coef.size

    # Inner loop function implementing the nested form of the Newton polynomial, which corresponds to Horner's scheme for Newton form
    def horner_step(i, val):
        reverse_index = n - 1 - i
        return val * (evaluation_points - interpolation_nodes[reverse_index]) + coef[reverse_index]

    # Initialize the array to hold the evaluated polynomial values
    polynomial_values = jnp.zeros_like(evaluation_points)

    # Evaluate the polynomial using Horner's method
    polynomial_values = jax.lax.fori_loop(0, n, horner_step, polynomial_values)

    # Return scalar if input was scalar, otherwise return array
    return polynomial_values if evaluation_points.ndim > 0 else polynomial_values.item()
