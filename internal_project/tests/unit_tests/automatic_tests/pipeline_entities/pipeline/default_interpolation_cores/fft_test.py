######################################################
# TODO! This class has not been tested yet since the #
#       interpolant is not implemented yet           #
######################################################

import unittest
from unittest.mock import MagicMock
import importlib

import jax.numpy as jnp

from data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.pipeline.component_entities.default_components.default_interpolation_cores.fft_interpolation_core import (
    FFTInterpolationCore,
)

# We also import the module itself so we can monkey-patch its symbol
fft_core_module = importlib.import_module(
    "pipeline_entities.pipeline.component_entities.default_component_types.interpolation_cores.fft_interpolation_core"
)


class _DummyFFTInterpolant:
    """Test double to capture what the core passes to the interpolant."""
    def __init__(self, nodes, weights, interval):
        self.nodes = nodes
        self.weights = weights
        self.interval = interval

    # Provide a minimal repr/eq if you like (not required for this test)


class TestFFTInterpolationCore(unittest.TestCase):
    def setUp(self):
        # Patch the interpolant used inside the FFT core
        self._orig_interpolant = fft_core_module.FastFourierTransformationInterpolant
        fft_core_module.FastFourierTransformationInterpolant = _DummyFFTInterpolant

    def tearDown(self):
        # Restore original symbol
        fft_core_module.FastFourierTransformationInterpolant = self._orig_interpolant

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
    def _pipeline_data(nodes: jnp.ndarray, values: jnp.ndarray, interval: jnp.ndarray, dtype) -> PipelineData:
        pd = PipelineData()
        pd.data_type = dtype
        pd.interpolation_nodes = nodes.astype(dtype)
        pd.interpolation_values = values.astype(dtype)
        pd.interpolation_interval = interval.astype(dtype)
        pd.interpolant = None
        return pd

    def test_fft_weights_and_plumbing_float32(self):
        dtype = jnp.float32
        N = 8

        # Use a simple periodic sample so the FFT is deterministic
        # e.g., f_k = sin(2πk/N) + 0.5*cos(4πk/N)
        k = jnp.arange(N, dtype=dtype)
        values = jnp.sin(2.0 * jnp.pi * k / N) + 0.5 * jnp.cos(4.0 * jnp.pi * k / N)

        # Nodes & interval are passed through but not used by FFT; still verify plumbing
        nodes = jnp.linspace(-1.0, 1.0, N, dtype=dtype)
        interval = jnp.array([-1.0, 1.0], dtype=dtype)

        pd = self._pipeline_data(nodes, values, interval, dtype)
        core = FFTInterpolationCore([pd], self._exec_data())

        # Act
        result_pd = core.perform_action()

        # Plumbing: same object, interpolant is our dummy
        self.assertIs(result_pd, pd)
        self.assertIsInstance(pd.interpolant, _DummyFFTInterpolant)

        # The core should pass through nodes and interval unchanged
        self.assertTrue(jnp.array_equal(pd.interpolant.nodes, nodes))
        self.assertTrue(jnp.array_equal(pd.interpolant.interval, interval))

        # Weights must be FFT(values)/N (complex output)
        expected_weights = jnp.fft.fft(values) / N
        self.assertTrue(jnp.allclose(pd.interpolant.weights, expected_weights, rtol=1e-6, atol=0.0))


if __name__ == "__main__":
    unittest.main()
