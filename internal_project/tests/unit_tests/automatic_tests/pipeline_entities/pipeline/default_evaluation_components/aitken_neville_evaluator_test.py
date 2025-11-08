import unittest
import jax
import jax.numpy as jnp

from pipeline_entities.pipeline.component_entities.default_components.default_evaluation_components.aitken_neville_evaluator import (
    AitkenNevilleEvaluator,
)


class _FakePipelineData:
    def __init__(self, nodes, values, eval_points, dtype=jnp.float32):
        self.data_type = dtype
        self.interpolation_nodes = jnp.asarray(nodes, dtype=dtype)
        self.interpolation_values = jnp.asarray(values, dtype=dtype)
        self.interpolant_evaluation_points = jnp.asarray(eval_points, dtype=dtype)
        self.node_count = int(self.interpolation_nodes.size)
        self.interpolant_values = None


def _make_component(nodes, values, eval_points, dtype=jnp.float32):
    """
    Create evaluator instance with fake PipelineData and a **zero-arg** jitted
    bound method (matching the component's current behavior).
    """
    pd = _FakePipelineData(nodes, values, eval_points, dtype=dtype)
    comp = AitkenNevilleEvaluator.__new__(AitkenNevilleEvaluator)
    comp._pipeline_data_ = [pd]

    comp._compiled_jax_callable_ = jax.jit(comp._internal_perform_action_)
    return comp, pd


def _tol_from_dtype(x):
    eps = jnp.finfo(jnp.asarray(x).dtype).eps
    # multipliers so float32 passes
    rtol = 1000.0 * float(eps)
    atol = 1000.0 * float(eps)
    return rtol, atol


def _x64_enabled():
    return jnp.array(0.0, dtype=jnp.float64).dtype == jnp.float64


class TestAitkenNevilleEvaluator(unittest.TestCase):
    def test_linear(self):
        # p(x) = 2 + 3x
        p = lambda x: 2.0 + 3.0 * x
        nodes = jnp.array([0.0, 1.0])
        values = p(nodes)
        xs = jnp.array([-0.5, 0.0, 0.25, 1.0, 2.0])

        comp, pd = _make_component(nodes, values, xs, dtype=jnp.float32)
        out_pd = comp.perform_action()
        y_hat = out_pd.interpolant_values

        rtol, atol = _tol_from_dtype(y_hat)
        self.assertEqual(y_hat.shape, xs.shape)
        self.assertTrue(jnp.allclose(y_hat, p(xs), rtol=rtol, atol=atol))

    def test_quadratic(self):
        # p(x) = 1 - 2x + x**2
        def p(x):
            return 1.0 - 2.0 * x + x**2

        nodes = jnp.array([-1.0, 0.0, 2.0])
        values = p(nodes)
        xs = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.5, 2.0])

        comp, pd = _make_component(nodes, values, xs, dtype=jnp.float32)
        out_pd = comp.perform_action()
        y_hat = out_pd.interpolant_values

        rtol, atol = _tol_from_dtype(y_hat)
        self.assertEqual(y_hat.shape, xs.shape)
        self.assertTrue(jnp.allclose(y_hat, p(xs), rtol=rtol, atol=atol))

    def test_dtype_respected(self):
        p = lambda x: x**3 - x + 1.0
        nodes = jnp.array([-1.0, 0.5, 1.2, 2.0])
        values = p(nodes)
        xs = jnp.linspace(-1.0, 2.0, 7)

        comp32, pd32 = _make_component(nodes, values, xs, dtype=jnp.float32)
        out_pd32 = comp32.perform_action()
        self.assertEqual(out_pd32.interpolant_values.dtype, jnp.float32)

        if _x64_enabled():
            comp64, pd64 = _make_component(nodes, values, xs, dtype=jnp.float64)
            out_pd64 = comp64.perform_action()
            self.assertEqual(out_pd64.interpolant_values.dtype, jnp.float64)


if __name__ == "__main__":
    unittest.main()
