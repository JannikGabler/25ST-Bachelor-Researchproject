# tests/common/test_interpolation_core_base.py
import unittest
from unittest.mock import MagicMock
import jax.numpy as jnp
from data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData


class TestInterpolationCoreBase(unittest.TestCase):
    """
    Extend this with:
      - CORE_CLS: the interpolation core class (callable as CORE_CLS([pd], exec_data))
      - build_repr(nodes, values): returns the 'expected representation' (e.g., coeffs or weights)
      - eval_with_repr(xs, nodes, values, repr_): evaluates polynomial using the representation
      - INTERPOLANT_CLS: the interpolant type expected to be attached
    """

    CORE_CLS = None
    INTERPOLANT_CLS = None

    # ---- Hooks each subclass must implement ----
    def build_repr(self, nodes: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray: ...
    def eval_with_repr(self, xs: jnp.ndarray, nodes: jnp.ndarray, values: jnp.ndarray, repr_: jnp.ndarray) -> jnp.ndarray: ...

    # ---- Common helpers ----
    @staticmethod
    def _exec_data():
        return AdditionalComponentExecutionData(
            overridden_attributes={},
            pipeline_configuration=MagicMock(),
            pipeline_input=MagicMock(),
            own_graph_node=MagicMock(),
            component_execution_reports=[],
        )

    @staticmethod
    def _pd(nodes, values, dtype):
        pd = PipelineData()
        pd.data_type = dtype
        pd.node_count = nodes.size
        pd.interpolation_nodes = nodes.astype(dtype)
        pd.interpolation_values = values.astype(dtype)
        pd.interpolant = None
        return pd

    def _run_case(self, nodes, values, f_true, xs, rtol=1e-6):
        self.assertIsNotNone(self.CORE_CLS, "Subclass must set CORE_CLS")
        self.assertIsNotNone(self.INTERPOLANT_CLS, "Subclass must set INTERPOLANT_CLS")

        dtype = jnp.float32  # portable (no x64 requirement)
        pd = self._pd(nodes.astype(dtype), values.astype(dtype), dtype)

        # Act
        core = self.CORE_CLS([pd], self._exec_data())
        result_pd = core.perform_action()

        # Plumbing
        self.assertIs(result_pd, pd)
        self.assertIsNotNone(pd.interpolant)
        self.assertIsInstance(pd.interpolant, self.INTERPOLANT_CLS)

        # Representation matches expected
        expected_repr = self.build_repr(nodes.astype(dtype), values.astype(dtype)).astype(dtype)

        # grab actual repr from interpolant (each subclass knows where to read it)
        actual_repr = self._extract_repr_from_interpolant(pd.interpolant).astype(dtype)
        self.assertTrue(jnp.allclose(expected_repr, actual_repr, rtol=rtol, atol=0.0))

        # Numeric correctness (evaluate using the repr)
        expected = f_true(xs.astype(dtype))
        actual = self.eval_with_repr(xs.astype(dtype), nodes.astype(dtype), values.astype(dtype), actual_repr)
        self.assertTrue(jnp.allclose(expected, actual, rtol=rtol, atol=0.0))

    # Subclasses can override if their interpolant stores repr differently
    def _extract_repr_from_interpolant(self, interpolant):
        # Default assumes "_coefficients_" (Aitken–Neville case)
        return interpolant._coefficients_
