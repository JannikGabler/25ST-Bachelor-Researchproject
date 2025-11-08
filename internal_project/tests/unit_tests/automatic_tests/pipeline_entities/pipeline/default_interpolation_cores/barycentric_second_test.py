import unittest
import jax.numpy as jnp

from test_interpolation_core_base import TestInterpolationCoreBase
from functions.defaults.default_interpolants.barycentric_second_interpolant import (
    BarycentricSecondInterpolant,
)
from pipeline_entities.pipeline.component_entities.default_components.default_interpolation_cores.barycentric_second_interpolation_core import (
    BarycentricSecondInterpolationCore,
)


class TestBarycentricSecondInterpolationCore(TestInterpolationCoreBase):
    CORE_CLS = BarycentricSecondInterpolationCore
    INTERPOLANT_CLS = BarycentricSecondInterpolant

    # The representation for this core is also the barycentric weights.
    def _extract_repr_from_interpolant(self, interpolant):
        return interpolant._weights_

    def build_repr(self, nodes: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
        """
        Standard barycentric weights (work for first & second form):
            w_j = 1 / Π_{k≠j} (x_j - x_k)
        """
        n = nodes.size
        w = jnp.empty_like(nodes)
        for j in range(n):
            diffs = nodes[j] - nodes
            diffs = diffs.at[j].set(1.0)  # ignore self
            w = w.at[j].set(1.0 / jnp.prod(diffs))
        return w

    def eval_with_repr(
        self,
        xs: jnp.ndarray,
        nodes: jnp.ndarray,
        values: jnp.ndarray,
        weights: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Second form:
            p(x) =  Σ_j w_j f_j / (x - x_j)  /  Σ_j w_j / (x - x_j)
        Return exact f_j if x equals a node.
        """

        def eval_one(x):
            diffs = x - nodes
            exact_mask = diffs == 0.0
            if bool(jnp.any(exact_mask)):
                idx = int(jnp.argmax(exact_mask))
                return values[idx]
            num = jnp.sum(weights * values / diffs)
            den = jnp.sum(weights / diffs)
            return num / den

        return jnp.array([eval_one(x) for x in xs], dtype=values.dtype)

    # ---- Concrete cases ----

    def test_linear(self):
        # p(x) = 2 + 3x  -> nodes [0,1], values [2,5]
        nodes = jnp.array([0.0, 1.0])
        values = jnp.array([2.0, 5.0])
        f_true = lambda x: 2.0 + 3.0 * x
        xs = jnp.array([0.0, 0.2, 1.0, 1.5])
        self._run_case(nodes, values, f_true, xs)

    def test_quadratic(self):
        # p(x) = 1 + 2x + x^2 -> nodes [0,1,2], values [1,4,9]
        nodes = jnp.array([0.0, 1.0, 2.0])
        values = (nodes**2) + 2.0 * nodes + 1.0
        f_true = lambda x: (x**2) + 2.0 * x + 1.0
        xs = jnp.array([-1.0, 0.0, 0.5, 1.0, 3.0])
        self._run_case(nodes, values, f_true, xs)


if __name__ == "__main__":
    unittest.main()
