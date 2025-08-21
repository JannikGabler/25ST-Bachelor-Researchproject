import unittest
from unittest.mock import MagicMock

import jax.numpy as jnp

from interpolants.default_interpolants.barycentric_first_interpolant import BarycentricFirstInterpolant
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.pipeline.component_entities.default_components.default_interpolation_cores.barycentric_first_interpolation_core import (
    BarycentricFirstInterpolationCore,
)


class TestBarycentricFirstInterpolationCore(unittest.TestCase):
    @staticmethod
    def _dummy_execution_data() -> AdditionalComponentExecutionData:
        # Satisfy required ctor args with safe stand-ins
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
    def _expected_barycentric_first_weights(nodes: jnp.ndarray) -> jnp.ndarray:
        """
        w_j = 1 / prod_{k != j} (x_j - x_k)
        """
        n = nodes.size
        w = jnp.empty_like(nodes)
        for j in range(n):
            diffs = nodes[j] - nodes
            diffs = diffs.at[j].set(1.0)  # ignore self
            w = w.at[j].set(1.0 / jnp.prod(diffs))
        return w

    @staticmethod
    def _barycentric_first_eval(xs: jnp.ndarray, nodes: jnp.ndarray, values: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        """
        First form:
            p(x) = l(x) * sum_j (w_j * f_j / (x - x_j)),
        where l(x) = prod_j (x - x_j).
        If x equals a node, return the exact f_j.
        """
        def eval_one(x):
            diffs = x - nodes
            exact_mask = diffs == 0.0
            # exact node: return f_j
            if bool(jnp.any(exact_mask)):
                idx = int(jnp.argmax(exact_mask))
                return values[idx]
            l = jnp.prod(diffs)
            return l * jnp.sum(weights * values / diffs)

        # vectorized by python loop (tiny arrays in tests; OK for unit testing)
        return jnp.array([eval_one(x) for x in xs], dtype=values.dtype)

    def test_linear_weights_and_evaluation_float32(self):
        # p(x) = 2 + 3x; nodes at 0 and 1 => values 2 and 5; weights [-1, 1]
        dtype = jnp.float32
        nodes = jnp.array([0.0, 1.0], dtype=dtype)
        values = jnp.array([2.0, 5.0], dtype=dtype)

        pipeline_data = self._build_pipeline_data(nodes, values, dtype)
        core = BarycentricFirstInterpolationCore([pipeline_data], self._dummy_execution_data())

        result_pd = core.perform_action()

        # Plumbing
        self.assertIs(result_pd, pipeline_data)
        self.assertIsNotNone(result_pd.interpolant)
        self.assertIsInstance(result_pd.interpolant, BarycentricFirstInterpolant)

        # Weights match formula
        weights = result_pd.interpolant._weights_.astype(dtype)
        expected_w = self._expected_barycentric_first_weights(nodes)
        self.assertTrue(jnp.allclose(weights, expected_w, rtol=1e-6, atol=0.0))

        # Manual evaluation via first form equals ground truth
        xs = jnp.array([0.0, 0.2, 1.0, 1.5], dtype=dtype)
        expected = 2.0 + 3.0 * xs
        actual = self._barycentric_first_eval(xs, nodes, values, weights)
        self.assertTrue(jnp.allclose(expected, actual, rtol=1e-6, atol=0.0))

    def test_quadratic_weights_and_evaluation_float32(self):
        # p(x) = 1 + 2x + x^2; nodes [0,1,2] => values [1,4,9]; weights should be [0.5, -1.0, 0.5]
        dtype = jnp.float32
        nodes = jnp.array([0.0, 1.0, 2.0], dtype=dtype)
        values = (nodes ** 2) + 2.0 * nodes + 1.0  # [1, 4, 9]

        pipeline_data = self._build_pipeline_data(nodes, values, dtype)
        core = BarycentricFirstInterpolationCore([pipeline_data], self._dummy_execution_data())

        result_pd = core.perform_action()

        # Plumbing
        self.assertIs(result_pd, pipeline_data)
        self.assertIsInstance(result_pd.interpolant, BarycentricFirstInterpolant)

        # Weights
        weights = result_pd.interpolant._weights_.astype(dtype)
        expected_w = self._expected_barycentric_first_weights(nodes)
        self.assertTrue(jnp.allclose(weights, expected_w, rtol=1e-6, atol=0.0))

        # Manual evaluation via first form equals ground truth
        xs = jnp.array([-1.0, 0.0, 0.5, 1.0, 3.0], dtype=dtype)
        expected = (xs ** 2) + 2.0 * xs + 1.0
        actual = self._barycentric_first_eval(xs, nodes, values, weights)
        self.assertTrue(jnp.allclose(expected, actual, rtol=1e-6, atol=0.0))


if __name__ == "__main__":
    unittest.main()
