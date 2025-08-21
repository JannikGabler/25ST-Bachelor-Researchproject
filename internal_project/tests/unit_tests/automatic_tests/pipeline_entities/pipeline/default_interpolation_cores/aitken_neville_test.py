import jax.numpy as jnp
import unittest
from test_interpolation_core_base import TestInterpolationCoreBase
from interpolants.default_interpolants.aitken_neville_interpolant import AitkenNevilleInterpolant
from pipeline_entities.pipeline.component_entities.default_components.default_interpolation_cores.aitken_neville_interpolation_core import (
    AitkenNevilleInterpolationCore,
)

class TestAitkenNevilleCore(TestInterpolationCoreBase):
    CORE_CLS = AitkenNevilleInterpolationCore
    INTERPOLANT_CLS = AitkenNevilleInterpolant

    # Expected representation = monomial coefficients [c0, c1, ...]
    def build_repr(self, nodes, values):
        # Build coeffs for sanity cases (here we choose polynomials with exact samples)
        # For tests we pass in nodes/values already consistent with a polynomial; we just return
        # the known coefficients for those cases in each call site (see test methods below).
        raise NotImplementedError("Provide per-test known coeffs")

    def eval_with_repr(self, xs, nodes, values, coeffs):
        # Horner with ascending monomials
        acc = jnp.full_like(xs, coeffs[-1])
        for i in range(coeffs.size - 2, -1, -1):
            acc = acc * xs + coeffs[i]
        return acc

    # Override extraction (already good in base: _coefficients_)

    def test_linear(self):
        nodes = jnp.array([0., 1.])
        values = jnp.array([2., 5.])  # p(x)=2+3x
        coeffs = jnp.array([2., 3.])

        # quick closure that base class will call as ground truth
        f_true = lambda x: 2.0 + 3.0 * x

        # patch build_repr just for this case
        self.build_repr = lambda _n, _v: coeffs

        xs = jnp.array([0., 0.2, 1.0])
        self._run_case(nodes, values, f_true, xs)

    def test_quadratic(self):
        nodes = jnp.array([0., 1., 2.])
        values = (nodes**2) + 2*nodes + 1  # p(x)=1+2x+x^2
        coeffs = jnp.array([1., 2., 1.])
        f_true = lambda x: (x**2) + 2*x + 1
        self.build_repr = lambda _n, _v: coeffs
        xs = jnp.array([-1., 0., 0.5, 1., 3.])
        self._run_case(nodes, values, f_true, xs)


if __name__ == "__main__":
    unittest.main()