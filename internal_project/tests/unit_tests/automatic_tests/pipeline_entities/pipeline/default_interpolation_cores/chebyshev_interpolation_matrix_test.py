import unittest
import jax
import jax.numpy as jnp

from pipeline_entities.pipeline.component_entities.default_components.default_interpolation_cores.chebyshev_interpolation_matrix_core import (
    ChebyshevInterpolationMatrixCore,
)


class _FakePipelineData:
    def __init__(self, nodes: jnp.ndarray, data_type=jnp.float64):
        self.data_type = data_type
        self.interpolation_nodes = nodes.astype(data_type)
        self.node_count = nodes.size
        self.interpolation_matrix = None


class TestChebyshevInterpolationMatrixCore(unittest.TestCase):
    def _make_core_with_fake_pd(self, nodes: jnp.ndarray) -> tuple[ChebyshevInterpolationMatrixCore, _FakePipelineData]:
        pd = _FakePipelineData(nodes)
        # Bypass heavy base-class init: allocate object & inject required attrs
        core = ChebyshevInterpolationMatrixCore.__new__(ChebyshevInterpolationMatrixCore)
        core._pipeline_data_ = [pd]

        # Use the internal (JAX) function as the compiled callable (mirrors AOTC pattern)
        internal = ChebyshevInterpolationMatrixCore._internal_perform_action_(core)
        # In the class, _internal_perform_action_ uses self._pipeline_data_; we need a closure/callable:
        def _call():
            return ChebyshevInterpolationMatrixCore._internal_perform_action_(core)

        # Optionally jit-compile to mimic production
        core._compiled_jax_callable_ = jax.jit(_call)
        return core, pd

    def test_matrix_matches_cos_definition(self):
        # Use n Chebyshev nodes in [-1,1]; any distinct nodes are fine for this test.
        # We'll take second-kind nodes including endpoints for clear angles.
        n = 5
        k = jnp.arange(n)
        nodes = jnp.cos(jnp.pi * k / (n - 1))  # U-kind grid (includes ±1)

        core, pd = self._make_core_with_fake_pd(nodes)
        # run action -> writes pd.interpolation_matrix
        out_pd = core.perform_action()
        M = out_pd.interpolation_matrix
        self.assertIsNotNone(M)
        self.assertEqual(M.shape, (n, n))

        # Ground truth: M[i, j] = cos(j * arccos(x_i))
        theta = jnp.arccos(jnp.clip(nodes, -1.0, 1.0))
        j = jnp.arange(n)
        # Broadcast to build (n, n)
        M_true = jnp.cos(theta[:, None] * j[None, :])

        self.assertTrue(jnp.allclose(M, M_true, atol=1e-12, rtol=1e-12))

    def test_reconstruct_known_polynomial(self):
        # Choose n=4 → degree up to 3. Build samples from a known cubic polynomial p(x)
        n = 4
        k = jnp.arange(n)
        nodes = jnp.cos(jnp.pi * k / (n - 1))  # include endpoints

        # True polynomial in monomials: p(x) = 1 - 2x + 0.5 x^2 + 3 x^3
        def p(x):
            return 1.0 - 2.0 * x + 0.5 * x**2 + 3.0 * x**3

        y = p(nodes)

        core, pd = self._make_core_with_fake_pd(nodes)
        out_pd = core.perform_action()
        M = out_pd.interpolation_matrix

        # Solve for Chebyshev coefficients c in T_j basis: M c = y
        # Use least squares in case nodes are not exactly square/ideal numerically
        c, *_ = jnp.linalg.lstsq(M, y, rcond=None)

        # Evaluate interpolant q(x)=sum_j c_j T_j(x) at a few test points and compare to p(x)
        xs = jnp.array([-1.0, -0.25, 0.0, 0.3, 1.0])
        theta = jnp.arccos(jnp.clip(xs, -1.0, 1.0))
        j_idx = jnp.arange(n)
        T = jnp.cos(theta[:, None] * j_idx[None, :])  # rows: x, cols: degree
        q = T @ c

        # Choose tolerances appropriate to dtype (float32 vs float64)
        eps = jnp.finfo(q.dtype).eps
        rtol = 1000 * eps
        atol = 1000 * eps

        self.assertTrue(jnp.allclose(q, p(xs), rtol=rtol, atol=atol))


if __name__ == "__main__":
    unittest.main()
