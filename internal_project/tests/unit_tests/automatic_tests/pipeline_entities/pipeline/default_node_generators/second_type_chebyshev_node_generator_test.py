import unittest
import jax.numpy as jnp

from test_node_generator_base import TestNodeGeneratorBase
from pipeline_entities.pipeline.component_entities.default_components.default_node_generators.second_type_chebyshev_node_generator import (
    SecondTypeChebyshevNodeGenerator,
)


class TestSecondTypeChebyshevNodeGenerator(TestNodeGeneratorBase):
    CORE_CLS = SecondTypeChebyshevNodeGenerator

    def build_expected_nodes(self, node_count: int, interval: jnp.ndarray, dtype) -> jnp.ndarray:
        """
        Chebyshev 2nd-kind nodes on [-1,1] (incl. endpoints) for n>=2:
          x_k = cos(pi*k/(n-1)), k = n-1, ..., 0
        For n=1: use midpoint 0 on [-1,1], then rescale to (a+b)/2.
        """
        a, b = interval[0], interval[1]

        if node_count == 1:
            mid = (a + b) / jnp.asarray(2, dtype=dtype)
            return jnp.array([mid], dtype=dtype)

        k = jnp.arange(node_count - 1, -1, -1, dtype=dtype)  # n-1, ..., 0 for ascending order
        denom = jnp.asarray(node_count - 1, dtype=dtype)
        base = jnp.cos(jnp.pi * k / denom).astype(dtype)

        if (a != jnp.asarray(-1, dtype=dtype)) or (b != jnp.asarray(1, dtype=dtype)):
            scale = (b - a) / jnp.asarray(2, dtype=dtype)
            return base * scale + (a + scale)
        return base

    def test_default_interval_float32(self):
        # [-1, 1], n=6
        self._run_case(node_count=6, interval=(-1.0, 1.0), dtype=jnp.float32)

    def test_rescaled_interval_float32(self):
        # Slightly looser tolerances due to float32 cos + rescale
        self._run_case(
            node_count=7,
            interval=(-2.5, 3.25),
            dtype=jnp.float32,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_two_nodes_endpoints(self):
        # n=2 -> ascending [-1, 1] on [-1,1], then rescale
        self._run_case(node_count=2, interval=(0.0, 10.0), dtype=jnp.float32)

    def test_single_node_supported(self):
        # n=1 -> single midpoint
        self._run_case(node_count=1, interval=(-3.0, 5.0), dtype=jnp.float32)
        self._run_case(node_count=1, interval=(-1.0, 1.0), dtype=jnp.float32)  # → [0.0]


if __name__ == "__main__":
    unittest.main()
