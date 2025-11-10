import jax.numpy as jnp
import unittest

from test_node_generator_base import TestNodeGeneratorBase
from pipeline_entities.pipeline.component_entities.default_components.default_node_generators.first_type_chebyshev_node_generator import (
    FirstTypeChebyshevNodeGenerator,
)


class TestFirstTypeChebyshevNodeGenerator(TestNodeGeneratorBase):
    CORE_CLS = FirstTypeChebyshevNodeGenerator

    def build_expected_nodes(
        self, node_count: int, interval: jnp.ndarray, dtype
    ) -> jnp.ndarray:
        """
        Implementation mirrors your generator:

        Raw Chebyshev (first kind, Gauss nodes; no endpoints):
          x_k = cos( (2k-1) * pi / (2n) ),  k=1..n, on [-1, 1]

        Rescale to [a, b] if interval != [-1, 1]:
          x' = ((b - a) / 2) * x + (a + b) / 2
        """
        # base nodes on [-1, 1]; ascending (left-to-right) order
        k = jnp.arange(node_count - 1, -1, -1, dtype=dtype)  # n-1, n-2, ..., 0
        thetas = (2 * k + 1) * (jnp.pi / (2 * node_count))
        base = jnp.cos(thetas).astype(dtype)

        a, b = interval[0], interval[1]
        if (a != jnp.asarray(-1, dtype=dtype)) or (b != jnp.asarray(1, dtype=dtype)):
            scale = (b - a) / jnp.asarray(2, dtype=dtype)
            return base * scale + (a + scale)
        return base

    def test_default_interval_float32(self):
        # [-1, 1], n=6
        self._run_case(node_count=6, interval=(-1.0, 1.0), dtype=jnp.float32)

    def test_rescaled_interval_float32(self):
        # rescale to arbitrary [a, b]
        self._run_case(node_count=7, interval=(-2.5, 3.0), dtype=jnp.float32)

    def test_single_node_edge_case(self):
        # n=1 -> cos(pi/2) = 0 on [-1,1]; after rescale it's midpoint (a+b)/2
        self._run_case(node_count=1, interval=(-3.0, 5.0), dtype=jnp.float32)


if __name__ == "__main__":
    unittest.main()
