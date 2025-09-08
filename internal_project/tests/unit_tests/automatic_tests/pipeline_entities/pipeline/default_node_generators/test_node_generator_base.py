# test_node_generator_base.py
import unittest
from unittest.mock import MagicMock
import jax.numpy as jnp

from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import (
    AdditionalComponentExecutionData,
)


class TestNodeGeneratorBase(unittest.TestCase):
    """
    Reusable base for NodeGenerator tests.

    Subclasses must set:
      - CORE_CLS: the node generator class (called as CORE_CLS([pd], exec_data))
      - build_expected_nodes(node_count, interval, dtype) -> jnp.ndarray

    Provided assertions:
      - plumbing: perform_action returns same PipelineData
      - exact node values vs. expected
      - dtype, length, and bounds checks
    """

    CORE_CLS = None  # override in child

    # hooks to implement in child
    def build_expected_nodes(self, node_count: int, interval: jnp.ndarray, dtype) -> jnp.ndarray:
        raise NotImplementedError

    @staticmethod
    def _exec_data() -> AdditionalComponentExecutionData:
        return AdditionalComponentExecutionData(
            overridden_attributes={},
            pipeline_configuration=MagicMock(),
            pipeline_input=MagicMock(),
            own_graph_node=MagicMock(),
            component_execution_reports=[],
        )

    @staticmethod
    def _pd(dtype, node_count: int, interval) -> PipelineData:
        pd = PipelineData()
        pd.data_type = dtype
        pd.node_count = node_count
        pd.interpolation_interval = jnp.array(interval, dtype=dtype)
        pd.interpolation_nodes = None
        return pd

    def _run_case(self, node_count: int, interval, dtype=jnp.float32, rtol=1e-7, atol=0.0):
        self.assertIsNotNone(self.CORE_CLS, "Subclass must set CORE_CLS")

        interval_arr = jnp.array(interval, dtype=dtype)
        pd = self._pd(dtype, node_count, interval_arr)
        core = self.CORE_CLS([pd], self._exec_data())

        # Act
        result_pd = core.perform_action()

        # Plumbing
        self.assertIs(result_pd, pd)
        self.assertIsNotNone(pd.interpolation_nodes)

        nodes = pd.interpolation_nodes
        expected = self.build_expected_nodes(node_count, interval_arr, dtype)

        # Basic checks
        self.assertEqual(nodes.dtype, dtype)
        self.assertEqual(nodes.size, node_count)

        # Values check
        self.assertTrue(jnp.allclose(nodes, expected, rtol=rtol, atol=atol))

        # Bounds check (Chebyshev may not include endpoints; just ensure within [a, b])
        a, b = float(interval_arr[0]), float(interval_arr[1])
        self.assertTrue(jnp.all(nodes >= min(a, b)))
        self.assertTrue(jnp.all(nodes <= max(a, b)))
