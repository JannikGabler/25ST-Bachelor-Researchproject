import jax.numpy as jnp
import unittest

from test_interpolation_core_base import TestInterpolationCoreBase
from functions.defaults.default_interpolants.barycentric_second_interpolant import BarycentricSecondInterpolant
from pipeline_entities.pipeline.component_entities.default_components.default_interpolation_cores.barycentric_second_chebyshev_interpolation_core import (
    BarycentricSecondChebyshevInterpolationCore,
)

############################################
# TODO! This class has not been tested yet #
############################################

class TestBarycentricSecondChebyshevInterpolationCore(TestInterpolationCoreBase):
    CORE_CLS = BarycentricSecondChebyshevInterpolationCore
    INTERPOLANT_CLS = BarycentricSecondInterpolant

    # The “representation” here is the Chebyshev barycentric weights (second form).
    def _extract_repr_from_interpolant(self, interpolant):
        return interpolant._weights_

    # ---- Helpers for this core ----
    @staticmethod
    def _chebyshev_nodes_first_kind(n: int, dtype=jnp.float32) -> jnp.ndarray:
        """
        n Chebyshev nodes of the first kind on [-1, 1], including endpoints:
          x_k = cos(pi * k / (n-1)),  k=0..n-1
        """
        k = jnp.arange(n, dtype=jnp.int32)
        return jnp.cos(jnp.pi * k.astype(dtype) / jnp.asarray(n - 1, dtype=dtype), dtype=dtype)

    def build_repr(self, nodes: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
        """
        Expected Chebyshev weights (standard choice):
          w_k = (-1)^k, with endpoints halved: w_0 /= 2, w_{n-1} /= 2
        Independent of the node values themselves (only depends on n and index).
        """
        n = nodes.size
        idx = jnp.arange(n, dtype=jnp.int32)
        w = jnp.where(idx % 2 == 0, 1.0, -1.0).astype(nodes.dtype)
        if n >= 1:
            w = w.at[0].set(w[0] * 0.5)
        if n >= 2:
            w = w.at[-1].set(w[-1] * 0.5)
        return w

    def eval_with_repr(
        self,
        xs: jnp.ndarray,
        nodes: jnp.ndarray,
        values: jnp.ndarray,
        weights: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Barycentric second form:
          p(x) =  (Σ w_j f_j / (x - x_j)) / (Σ w_j / (x - x_j))
        Return exact f_j if x == x_j.
        """
        def eval_one(x):
            diffs = x - nodes
            exact = diffs == 0.0
            if bool(jnp.any(exact)):
                idx = int(jnp.argmax(exact))
                return values[idx]
            num = jnp.sum(weights * values / diffs)
            den = jnp.sum(weights / diffs)
            return num / den

        return jnp.array([eval_one(x) for x in xs], dtype=values.dtype)

    # ---- Concrete cases ----

    def test_linear(self):
        # p(x) = 2 + 3x over Chebyshev nodes (n=2 -> nodes = [cos(0), cos(pi)] = [1, -1])
        n = 2
        nodes = self._chebyshev_nodes_first_kind(n)
        values = 2.0 + 3.0 * nodes
        f_true = lambda x: 2.0 + 3.0 * x
        xs = jnp.array([-1.0, -0.25, 0.0, 0.7, 1.0], dtype=jnp.float32)
        self._run_case(nodes, values, f_true, xs)

    def test_quadratic(self):
        # p(x) = 1 + 2x + x^2 over Chebyshev nodes (n=3 -> nodes = [1, 0, -1])
        n = 3
        nodes = self._chebyshev_nodes_first_kind(n)
        values = (nodes ** 2) + 2.0 * nodes + 1.0
        f_true = lambda x: (x ** 2) + 2.0 * x + 1.0
        xs = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=jnp.float32)
        self._run_case(nodes, values, f_true, xs)


if __name__ == "__main__":
    unittest.main()
