import unittest
import jax.numpy as jnp

from test_project.src.newton import divided_differences, newton_interpolate


class MyTestCase(unittest.TestCase):
    def test_exact_interpolation_points(self):
        x_data = jnp.array([0.0, 1.0, 2.0])
        y_data = x_data ** 2 + 1
        coef = divided_differences(x_data, y_data)

        for x, y in zip(x_data, y_data):
            y_interp = newton_interpolate(x, x_data, coef)
            self.assertTrue(jnp.isclose(y_interp, y, atol=1e-8))

    def test_quadratic_function_midpoint(self):
        x_data = jnp.array([0.0, 1.0, 2.0])
        y_data = x_data ** 2 + 1
        coef = divided_differences(x_data, y_data)

        x_test = 1.5
        expected = x_test ** 2 + 1
        y_interp = newton_interpolate(x_test, x_data, coef)
        self.assertTrue(jnp.isclose(y_interp, expected, atol=1e-6))

    def test_constant_function(self):
        x_data = jnp.array([-2.0, 0.0, 2.0])
        y_data = jnp.array([5.0, 5.0, 5.0])
        coef = divided_differences(x_data, y_data)

        x_test = jnp.linspace(-2, 2, 5)
        y_interp = newton_interpolate(x_test, x_data, coef)
        self.assertTrue(jnp.allclose(y_interp, 5.0))

    def test_linear_function(self):
        x_data = jnp.array([1.0, 2.0, 3.0])
        y_data = 3 * x_data + 2
        coef = divided_differences(x_data, y_data)

        x_test = jnp.array([1.5, 2.5])
        y_expected = 3 * x_test + 2
        y_interp = newton_interpolate(x_test, x_data, coef)
        self.assertTrue(jnp.allclose(y_interp, y_expected, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
