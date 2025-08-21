import unittest
from unittest.mock import MagicMock

import jax.numpy as jnp

from interpolants.default_interpolants.aitken_neville_interpolant import AitkenNevilleInterpolant
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.pipeline.component_entities.default_components.default_interpolation_cores.aitken_neville_interpolation_core import (
    AitkenNevilleInterpolationCore,
)


class TestAitkenNevilleInterpolationCore(unittest.TestCase):
    @staticmethod
    def _dummy_execution_data() -> AdditionalComponentExecutionData:
        # Satisfy ctor with harmless stand-ins
        return AdditionalComponentExecutionData(
            overridden_attributes={},
            pipeline_configuration=MagicMock(),
            pipeline_input=MagicMock(),
            own_graph_node=MagicMock(),
            component_execution_reports=[],
        )

    def _build_pipeline_data(
        self,
        nodes: jnp.ndarray,
        values: jnp.ndarray,
        dtype,
    ) -> PipelineData:
        pd = PipelineData()
        pd.data_type = dtype
        pd.node_count = nodes.size
        pd.interpolation_nodes = nodes.astype(dtype)
        pd.interpolation_values = values.astype(dtype)
        pd.interpolant = None
        return pd

    @staticmethod
    def _horner_eval(coeffs: jnp.ndarray, xs: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate polynomial in Horner form with ASCENDING monomial coefficients:
            coeffs = [c0, c1, c2, ...] means c0 + c1*x + c2*x^2 + ...
        """
        acc = jnp.full_like(xs, coeffs[-1])
        # iterate backwards through remaining coefficients
        for i in range(coeffs.size - 2, -1, -1):
            acc = acc * xs + coeffs[i]
        return acc

    def test_linear_coefficients_and_evaluation_float32(self):
        # Ground truth: p(x) = 2 + 3x  -> coefficients [2, 3]
        dtype = jnp.float32
        nodes = jnp.array([0.0, 1.0], dtype=dtype)
        values = jnp.array([2.0, 5.0], dtype=dtype)

        pipeline_data = self._build_pipeline_data(nodes, values, dtype)
        core = AitkenNevilleInterpolationCore([pipeline_data], self._dummy_execution_data())

        result_pd = core.perform_action()

        # Basic plumbing checks
        self.assertIs(result_pd, pipeline_data)
        self.assertIsNotNone(result_pd.interpolant)
        self.assertIsInstance(result_pd.interpolant, AitkenNevilleInterpolant)

        # Coefficients should be [intercept, slope] = [2, 3]
        coeffs = result_pd.interpolant._coefficients_.astype(dtype)
        self.assertTrue(jnp.array_equal(coeffs, jnp.array([2.0, 3.0], dtype=dtype)))

        # Evaluate with our own Horner (avoids needing interpolant._data_type_)
        xs = jnp.array([0.0, 0.2, 1.0], dtype=dtype)
        expected = 2.0 + 3.0 * xs
        actual = self._horner_eval(coeffs, xs)
        self.assertTrue(jnp.allclose(expected, actual, rtol=1e-6, atol=0.0))

    def test_quadratic_coefficients_and_evaluation_float32(self):
        # Ground truth: p(x) = 1 + 2x + x^2 -> coefficients [1, 2, 1]
        dtype = jnp.float32
        nodes = jnp.array([0.0, 1.0, 2.0], dtype=dtype)
        values = (nodes ** 2) + 2.0 * nodes + 1.0  # [1, 4, 9]

        pipeline_data = self._build_pipeline_data(nodes, values, dtype)
        core = AitkenNevilleInterpolationCore([pipeline_data], self._dummy_execution_data())

        result_pd = core.perform_action()

        # Basic plumbing checks
        self.assertIs(result_pd, pipeline_data)
        self.assertIsInstance(result_pd.interpolant, AitkenNevilleInterpolant)

        # Coefficients should be [1, 2, 1] (ascending monomial order)
        coeffs = result_pd.interpolant._coefficients_.astype(dtype)
        self.assertTrue(jnp.array_equal(coeffs, jnp.array([1.0, 2.0, 1.0], dtype=dtype)))

        # Evaluate with our own Horner
        xs = jnp.array([-1.0, 0.0, 0.5, 1.0, 3.0], dtype=dtype)
        expected = (xs ** 2) + 2.0 * xs + 1.0
        actual = self._horner_eval(coeffs, xs)
        self.assertTrue(jnp.allclose(expected, actual, rtol=1e-6, atol=0.0))


if __name__ == "__main__":
    unittest.main()
