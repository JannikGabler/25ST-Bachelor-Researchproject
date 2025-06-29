import unittest
import jax.numpy as jnp
import time


from test_project.src.barycentric_second.barycentric_second import compute_weights, barycentric_type2_interpolate


class MyTestCase(unittest.TestCase):

    def test_exact_interpolation_points(self):
        interpolation_nodes = jnp.array([0.0, 1.0, 2.0])
        function_values = interpolation_nodes ** 2 + 1
        for node, true_value in zip(interpolation_nodes, function_values):
            interpolated_value = barycentric_type2_interpolate(jnp.array([node]), interpolation_nodes, function_values)[0]
            self.assertTrue(jnp.isclose(interpolated_value, true_value, atol=1e-8),
                            msg=f"Expected {true_value}, got {interpolated_value}")

    def test_quadratic_function_midpoint(self):
        interpolation_nodes = jnp.array([0.0, 1.0, 2.0])
        function_values = interpolation_nodes ** 2 + 1
        evaluation_point = 1.5
        expected_value = evaluation_point ** 2 + 1
        interpolated_value = barycentric_type2_interpolate(jnp.array([evaluation_point]),
                                                           interpolation_nodes, function_values)[0]
        self.assertTrue(jnp.isclose(interpolated_value, expected_value, atol=1e-6),
                        msg=f"Expected {expected_value}, got {interpolated_value}")

    def test_constant_function(self):
        interpolation_nodes = jnp.array([-2.0, 0.0, 2.0])
        function_values = jnp.array([3.0, 3.0, 3.0])
        evaluation_points = jnp.linspace(-2, 2, 5)
        interpolated_values = barycentric_type2_interpolate(evaluation_points, interpolation_nodes, function_values)
        self.assertTrue(jnp.allclose(interpolated_values, 3.0),
                        msg=f"Expected all values to be 3.0, got {interpolated_values}")

    def test_linear_function(self):
        interpolation_nodes = jnp.array([1.0, 2.0, 3.0])
        function_values = 2 * interpolation_nodes + 5
        evaluation_points = jnp.array([1.5, 2.5])
        expected_values = 2 * evaluation_points + 5
        interpolated_values = barycentric_type2_interpolate(evaluation_points, interpolation_nodes, function_values)
        self.assertTrue(jnp.allclose(interpolated_values, expected_values, atol=1e-6),
                        msg=f"Expected {expected_values}, got {interpolated_values}")

    def test_evaluation_at_all_nodes(self):
        interpolation_nodes = jnp.array([-1.0, 0.0, 1.0])
        function_values = jnp.sin(interpolation_nodes)
        interpolated_values = barycentric_type2_interpolate(interpolation_nodes, interpolation_nodes, function_values)
        self.assertTrue(jnp.allclose(interpolated_values, function_values, atol=1e-8),
                        msg=f"Expected {function_values}, got {interpolated_values}")

    def test_interpolation_speed_various_sizes(self):
        grid_sizes = [200, 1000, 10000, 100000, 1000000]

        for n in grid_sizes:
            interpolation_nodes = jnp.linspace(-1.0, 1.0, 1000)
            function_values = interpolation_nodes ** 2
            evaluation_points = jnp.linspace(-1.0, 1.0, n)

            # Warm-up JIT compile
            barycentric_type2_interpolate(evaluation_points[:1], interpolation_nodes, function_values).block_until_ready()

            start = time.perf_counter()
            barycentric_type2_interpolate(evaluation_points, interpolation_nodes, function_values).block_until_ready()
            end = time.perf_counter()

            elapsed = end - start
            elapsed_ms = elapsed * 1000

            print(f"Type 2: Interpolation time for {n} evaluation points (JAX): {elapsed:.4f} seconds ({elapsed_ms:.1f} ms)")



if __name__ == '__main__':
    unittest.main()
