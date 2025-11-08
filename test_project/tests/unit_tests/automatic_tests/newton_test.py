import unittest
import jax.numpy as jnp
import time

from test_project.src.newton_interpolation.newton import newton_interpolate
from test_project.src.newton_interpolation.newton import divided_differences


class MyTestCase(unittest.TestCase):
    def test_constant_function_coefficients(self):
        interpolation_nodes = jnp.array([0.0, 1.0, 2.0])
        function_values = jnp.array([5.0, 5.0, 5.0])
        coeffs = divided_differences(interpolation_nodes, function_values)
        expected = jnp.array([5.0, 0.0, 0.0])
        self.assertTrue(jnp.allclose(coeffs, expected, atol=1e-8))

    def test_linear_function_coefficients(self):
        interpolation_nodes = jnp.array([0.0, 1.0, 2.0])
        function_values = 3 * interpolation_nodes + 2
        coeffs = divided_differences(interpolation_nodes, function_values)
        expected = jnp.array([2.0, 3.0, 0.0])
        self.assertTrue(jnp.allclose(coeffs, expected, atol=1e-8))

    def test_quadratic_function_coefficients(self):
        interpolation_nodes = jnp.array([0.0, 1.0, 2.0])
        function_values = interpolation_nodes**2 + 1
        coeffs = divided_differences(interpolation_nodes, function_values)
        expected = jnp.array([1.0, 1.0, 1.0])
        self.assertTrue(jnp.allclose(coeffs, expected, atol=1e-8))

    def test_exact_interpolation_points(self):
        interpolation_nodes = jnp.array([0.0, 1.0, 2.0])
        function_values = interpolation_nodes**2 + 1
        for node, true_value in zip(interpolation_nodes, function_values):
            interpolated_value = newton_interpolate(
                node, interpolation_nodes, function_values
            )
            self.assertTrue(jnp.isclose(interpolated_value, true_value, atol=1e-8))

    def test_quadratic_function_midpoint(self):
        interpolation_nodes = jnp.array([0.0, 1.0, 2.0])
        function_values = interpolation_nodes**2 + 1
        evaluation_point = 1.5
        expected_value = evaluation_point**2 + 1
        interpolated_value = newton_interpolate(
            evaluation_point, interpolation_nodes, function_values
        )
        self.assertTrue(jnp.isclose(interpolated_value, expected_value, atol=1e-6))

    def test_constant_function(self):
        interpolation_nodes = jnp.array([-2.0, 0.0, 2.0])
        function_values = jnp.array([5.0, 5.0, 5.0])
        evaluation_points = jnp.linspace(-2, 2, 5)
        interpolated_values = newton_interpolate(
            evaluation_points, interpolation_nodes, function_values
        )
        self.assertTrue(jnp.allclose(interpolated_values, 5.0))

    def test_linear_function(self):
        interpolation_nodes = jnp.array([1.0, 2.0, 3.0])
        function_values = 3 * interpolation_nodes + 2
        evaluation_points = jnp.array([1.5, 2.5])
        expected_values = 3 * evaluation_points + 2
        interpolated_values = newton_interpolate(
            evaluation_points, interpolation_nodes, function_values
        )
        self.assertTrue(jnp.allclose(interpolated_values, expected_values, atol=1e-6))

    def test_interpolation_speed_various_sizes(self):
        grid_sizes = [1000, 10000, 100000, 1000000]

        for n in grid_sizes:
            interpolation_nodes = jnp.linspace(-1.0, 1.0, 1000)
            function_values = interpolation_nodes**2
            evaluation_points = jnp.linspace(-1.0, 1.0, n)

            # Warm-up JIT compile
            newton_interpolate(
                evaluation_points[:1], interpolation_nodes, function_values
            ).block_until_ready()

            start = time.perf_counter()
            newton_interpolate(
                evaluation_points, interpolation_nodes, function_values
            ).block_until_ready()
            end = time.perf_counter()

            elapsed = end - start
            elapsed_ms = elapsed * 1000

            print(
                f"Interpolation time for {n} evaluation points (JAX): {elapsed:.4f} seconds ({elapsed_ms:.1f} ms)"
            )


if __name__ == "__main__":
    unittest.main()
