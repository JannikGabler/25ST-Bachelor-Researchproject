import unittest
import jax.numpy as jnp

from utils import jax_utils


class MyTestCase(unittest.TestCase):
    def test_rescale_interval_to_interval_1(self):
        old_interval = (-2, 1)
        new_interval = (0, 1)
        array = jnp.linspace(old_interval[0], old_interval[1], 5)

        expected = jnp.array([0, 1 / 4, 1 / 2, 3 / 4, 1])

        result = jax_utils.rescale_array_to_interval(array, old_interval, new_interval)

        rtol_array = jax_utils.relative_tolerances(expected, result)
        self.assertTrue(jnp.all(rtol_array <= 1e-05))

    def test_rescale_interval_to_interval_2(self):
        old_interval = (-1, 1)
        new_interval = (-2, 2)
        array = jnp.linspace(old_interval[0], old_interval[1], 5)

        expected = jnp.array([-2, -1, 0, 1, 2])

        result = jax_utils.rescale_array_to_interval(array, old_interval, new_interval)

        rtol_array = jax_utils.relative_tolerances(expected, result)
        self.assertTrue(jnp.all(rtol_array <= 1e-05))

    def test_rescale_interval_to_interval_3(self):
        old_interval = (-1, 1)
        new_interval = (-6, 1)
        array = jnp.linspace(old_interval[0], old_interval[1], 7, dtype=float)

        expected = jnp.array(
            [
                -6,
                -6 + 7 / 6,
                -6 + 2 * 7 / 6,
                -6 + 3 * 7 / 6,
                -6 + 4 * 7 / 6,
                -6 + 5 * 7 / 6,
                1,
            ]
        )

        result = jax_utils.rescale_array_to_interval(array, old_interval, new_interval)

        rtol_array = jax_utils.relative_tolerances(expected, result)
        self.assertTrue(jnp.all(rtol_array <= 1e-05))


if __name__ == "__main__":
    unittest.main()
