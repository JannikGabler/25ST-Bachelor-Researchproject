import unittest
from typing import Tuple
import jax.numpy as jnp

from pipeline.components.abstracts.node_generator import NodeGenerator
from pipeline.components.default_components.default_node_generators.equidistant_node_generator import EquidistantNodeGenerator
from pipeline.components.default_components.default_node_generators.first_type_chebyshev_node_generator import \
    FirstTypeChebyshevNodeGenerator


class MyTestCase(unittest.TestCase):
    def test_float16_generation_default_interval(self):
        interval: Tuple[float, float] = (-1, 1)
        node_count = 5
        data_type = jnp.float16
        generator: NodeGenerator = FirstTypeChebyshevNodeGenerator(interval, node_count, data_type)

        expected = jnp.array([jnp.float16("0.9510565163"), jnp.float16("0.5877852523"), jnp.float16("0"), jnp.float16("-0.5877852523"), jnp.float16("-0.9510565163")], dtype=jnp.float16)
        result = generator.generate_nodes()

        # Absolut tolerance is required because 0 appears as a value
        self.assertTrue(jnp.allclose(expected, result, atol=1E-3))



    def test_float16_generation_transformed_interval(self):
        interval: Tuple[float, float] = (-6, 1)
        node_count: int = 7
        data_type: type = jnp.float16
        generator: NodeGenerator = FirstTypeChebyshevNodeGenerator(interval, node_count, data_type)

        expected: jnp.ndarray = jnp.array([jnp.float16("0.9122476926"), jnp.float16("0.2364101886"),
                                        jnp.float16("-0.9814069131"), jnp.float16("-2.5"),
                                        jnp.float16("-4.018593087"), jnp.float16("-5.236410189"),
                                        jnp.float16("-5.912247693")], dtype=jnp.float16)
        result = generator.generate_nodes()

        self.assertTrue(jnp.allclose(expected, result, atol=1E-3))


    def test_float32_generation(self):
        interval: Tuple = (-1, 1)
        node_count = 4
        data_type = jnp.float32
        generator: EquidistantNodeGenerator = EquidistantNodeGenerator(interval, node_count, data_type)

        expected_nodes = jnp.array([jnp.float32("-1"), jnp.float32(-1/3), jnp.float32(1/3), jnp.float32("1")], dtype=jnp.float32)
        result = generator.generate_nodes()

        self.assertTrue(jnp.allclose(expected_nodes, result, rtol=1E-04))


if __name__ == '__main__':
    unittest.main()
