import jax.numpy as jnp
import unittest

from test_interpolation_core_base import TestInterpolationCoreBase
from functions.defaults.default_interpolants.newton_interpolant import NewtonInterpolant
from pipeline_entities.pipeline.component_entities.default_components.default_interpolation_cores.newton_interpolation_core import (
    NewtonInterpolationCore,
)


class TestNewtonInterpolationCore(TestInterpolationCoreBase):
    CORE_CLS = NewtonInterpolationCore
    INTERPOLANT_CLS = NewtonInterpolant

    # Representation = divided differences
    def _extract_repr_from_interpolant(self, interpolant):
        return interpolant._divided_differences_

    def build_repr(self, nodes: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
        """
        Build divided differences in-place so that the final vector is:
          [ f[x0], f[x0,x1], f[x0,x1,x2], ... ]
        matching the core’s output.
        """
        n = nodes.size
        c = values.copy()
        # standard in-place divided differences
        for j in range(1, n):
            for i in range(n - 1, j - 1, -1):
                c = c.at[i].set((c[i] - c[i - 1]) / (nodes[i] - nodes[i - j]))
        return c

    def eval_with_repr(
        self,
        xs: jnp.ndarray,
        nodes: jnp.ndarray,
        values: jnp.ndarray,
        dd: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Newton nested form (Horner-like for Newton basis):
          P(x) = c0 + c1*(x-x0) + c2*(x-x0)*(x-x1) + ...
        Implemented as:
          acc = 0
          for i = n-1..0:
              acc = acc * (x - nodes[i]) + dd[i]
        """

        def eval_one(x):
            acc = jnp.array(0.0, dtype=xs.dtype)
            for i in range(nodes.size - 1, -1, -1):
                acc = acc * (x - nodes[i]) + dd[i]
            return acc

        return jnp.array([eval_one(x) for x in xs], dtype=xs.dtype)

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
        values = (nodes**2) + 2.0 * nodes + 1.0  # [1,4,9]
        f_true = lambda x: (x**2) + 2.0 * x + 1.0
        xs = jnp.array([-1.0, 0.0, 0.5, 1.0, 3.0])
        self._run_case(nodes, values, f_true, xs)


if __name__ == "__main__":
    unittest.main()
