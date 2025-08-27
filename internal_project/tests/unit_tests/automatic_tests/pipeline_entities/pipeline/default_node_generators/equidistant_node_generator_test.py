import jax.numpy as jnp
import unittest

from test_node_generator_base import TestNodeGeneratorBase
from pipeline_entities.pipeline.component_entities.default_components.default_node_generators.equidistant_node_generator import (
    EquidistantNodeGenerator,
)


class TestEquidistantNodeGenerator(TestNodeGeneratorBase):
    CORE_CLS = EquidistantNodeGenerator

    def build_expected_nodes(self, node_count: int, interval: jnp.ndarray, dtype) -> jnp.ndarray:
        a, b = interval[0], interval[1]
        return jnp.linspace(a, b, node_count, dtype=dtype)

    def test_typical_interval_float32(self):
        # [-2, 3], n=5 → exact linspace
        self._run_case(node_count=5, interval=(-2.0, 3.0), dtype=jnp.float32)

    def test_single_node_edge_case(self):
        # n=1 → [a]
        self._run_case(node_count=1, interval=(-1.0, 1.0), dtype=jnp.float32)

    def test_decreasing_interval(self):
        # [b, a] with b < a → linspace handles decreasing endpoints
        self._run_case(node_count=6, interval=(1.5, -2.0), dtype=jnp.float32)

if __name__ == "__main__":
    unittest.main()
