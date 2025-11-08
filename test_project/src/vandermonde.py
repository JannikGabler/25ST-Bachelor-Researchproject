import jax
import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt


def vandermonde_interpolator(
    x_data: jnp.ndarray, y_data: jnp.ndarray, input_validation: bool = False
):
    """
    Given 1D arrays x_data, y_data of length n,
    returns a function p(x) evaluating the unique
    degree-(n-1) interpolant.

    !! Input validation is avoided and only meant for debugging purposes. It could impact performance !!

    If input validation is enabbled:
        Raises ValueError if:
        - x_data or y_data aren’t 1D
        - x_data and y_data differ in length
        - inputs are empty
        - x_data contains duplicate values
    """
    if input_validation:
        # Convert any array‐like to JAX arrays
        x_data = jnp.asarray(x_data)
        y_data = jnp.asarray(y_data)

        # 1) Must be 1D
        if x_data.ndim != 1 or y_data.ndim != 1:
            raise ValueError("x_data and y_data must be 1D arrays")

        # 2) Same length
        if x_data.shape[0] != y_data.shape[0]:
            raise ValueError("x_data and y_data must have the same length")

        n = x_data.shape[0]

        # 3) Non-empty
        if n == 0:
            raise ValueError("x_data and y_data must not be empty")

        # 4) Distinct x’s
        #    we use jnp.unique under the hood, but move to host for the length check
        if n != len(jnp.unique(x_data)):
            raise ValueError("x_data values must all be distinct")

    # build Vandermonde matrix (columns: x^0, x^1, ..., x^(n-1))
    V = jnp.vander(x_data, N=x_data.size, increasing=True)

    # solve for coefficients c (length n)
    c = jnp.linalg.solve(V, y_data)

    # 3) return a function that evaluates p(x) = sum c_j x^j
    @jax.jit  # Just-in-time compile for performance
    def p(x: jnp.ndarray):
        # For broadcasting, we can compute powers of x up to degree n-1:
        Xpow = jnp.vander(x, N=c.size, increasing=True)
        return Xpow @ c

    return p, c


def plot_interpolation(x_data, y_data, reference_func=None, num_points: int = 200):
    """
    Plot original data points, the interpolating polynomial curve,
    and optionally the expected (reference) function in another color.

    Args:
        x_data: sequence of x-coordinates (array-like).
        y_data: sequence of y-coordinates (array-like).
        reference_func: optional callable f(x) returning expected y.
        num_points: number of points to sample along the curve.
    """
    # Ensure arrays
    x = jnp.asarray(x_data)
    y = jnp.asarray(y_data)

    # Build interpolator
    p, _ = vandermonde_interpolator(x, y)

    # Sample evaluation points
    xs = jnp.linspace(x.min(), x.max(), num_points)
    ys = p(xs)

    # Convert to NumPy for plotting
    xs_np = np.array(xs)
    ys_np = np.array(ys)
    x_np = np.array(x)
    y_np = np.array(y)

    plt.figure()
    plt.scatter(x_np, y_np, label="Data points")
    plt.plot(xs_np, ys_np, label="Interpolation", linewidth=2)

    # Plot reference function if provided
    if reference_func is not None:
        y_ref_np = reference_func(xs_np)
        plt.plot(xs_np, y_ref_np, "--", label="Expected function", linewidth=2)

    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Example usage:
    # Interpolate f(x) = 2*x^2 + 3*x + 5
    def true_func(x):
        return 2 * x**2 + 3 * x + 5

    x_data = np.array([0.0, 1.0, 2.0, 4.0])
    y_data = true_func(x_data)
    # Pass the true function so it also gets plotted
    plot_interpolation(x_data, y_data, reference_func=true_func)
