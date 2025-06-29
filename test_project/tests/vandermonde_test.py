import pytest
import jax.numpy as jnp

import sys
from os.path import dirname, abspath, join

SRC_DIR = abspath(join(dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from vandermonde import vandermonde_interpolator


def test_known_polynomial_coefficients():
    # Polynomial: p(x) = 2*x^2 + 3*x + 5
    x_data = jnp.array([0.0, 1.0, 2.0])
    y_data = 2 * x_data**2 + 3 * x_data + 5
    # Get interpolator with validation on
    p, c = vandermonde_interpolator(x_data, y_data, input_validation=True)
    # Coefficients should be [5, 3, 2]
    expected_c = jnp.array([5.0, 3.0, 2.0])
    # Use jnp.allclose for comparison
    assert jnp.allclose(c, expected_c, atol=1e-6)

    # Test evaluation at new points
    xs = jnp.array([-1.0, 0.5, 3.0])
    ys = p(xs)
    expected_ys = 2 * xs**2 + 3 * xs + 5
    assert jnp.allclose(ys, expected_ys, atol=1e-6)


def test_input_validation_1d_requirement():
    # 2D input should raise ValueError
    x_data = jnp.array([[0.0, 1.0]])
    y_data = jnp.array([0.0, 1.0])
    with pytest.raises(ValueError):
        vandermonde_interpolator(x_data, y_data, input_validation=True)


def test_input_validation_length_mismatch():
    x_data = jnp.array([0.0, 1.0, 2.0])
    y_data = jnp.array([0.0, 1.0])
    with pytest.raises(ValueError):
        vandermonde_interpolator(x_data, y_data, input_validation=True)


def test_input_validation_empty():
    x_data = jnp.array([])
    y_data = jnp.array([])
    with pytest.raises(ValueError):
        vandermonde_interpolator(x_data, y_data, input_validation=True)


def test_input_validation_duplicates():
    x_data = jnp.array([1.0, 1.0, 2.0])
    y_data = jnp.array([2.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        vandermonde_interpolator(x_data, y_data, input_validation=True)


def test_interpolation_without_validation():
    # without validation, it should work for proper input
    x_data = jnp.array([0.0, 2.0])
    y_data = jnp.array([1.0, 5.0])  # linear p(x) = 2*x + 1
    p, c = vandermonde_interpolator(x_data, y_data, input_validation=False)
    expected_c = jnp.array([1.0, 2.0])
    assert jnp.allclose(c, expected_c, atol=1e-6)
    # Evaluate
    xs = jnp.linspace(0.0, 2.0, 5)
    ys = p(xs)
    expected_ys = 2 * xs + 1
    assert jnp.allclose(ys, expected_ys, atol=1e-6)
