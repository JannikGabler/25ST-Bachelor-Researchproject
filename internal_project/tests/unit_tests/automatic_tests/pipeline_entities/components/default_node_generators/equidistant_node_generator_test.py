import unittest
from typing import Tuple
import jax.numpy as jnp

from pipeline.components.default_components.default_node_generators.equidistant_node_generator import EquidistantNodeGenerator


class MyTestCase(unittest.TestCase):
    def test_float16_generation(self):
        interval: Tuple = (-1, 1)
        node_count = 5
        data_type = jnp.float16
        generator: EquidistantNodeGenerator = EquidistantNodeGenerator(interval, node_count, data_type)

        expected_nodes = jnp.array([jnp.float16("-1"), jnp.float16("-0.5"), jnp.float16("0"), jnp.float16("0.5"), jnp.float16("1")], dtype=jnp.float16)
        result = generator.generate_nodes()

        self.assertTrue(jnp.array_equal(expected_nodes, result))


    def test_float32_generation(self):
        interval: Tuple = (-1, 1)
        node_count = 4
        data_type = jnp.float16
        generator: EquidistantNodeGenerator = EquidistantNodeGenerator(interval, node_count, data_type)

        expected_nodes = jnp.array([jnp.float16("-1"), jnp.float16(-1/3), jnp.float16(1/3), jnp.float16("1")], dtype=jnp.float16)
        result = generator.generate_nodes()

        self.assertTrue(jnp.allclose(expected_nodes, result, rtol=1E-02))


if __name__ == '__main__':
    unittest.main()
