import jax
import jax.numpy as jnp

from interpolants.default_interpolants.fft_interpolant import FastFourierTransformationInterpolant
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.interpolation_cores.fft_interpolation_core_meta_info import \
    fft_interpolation_core_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import \
    pipeline_component
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import \
    AdditionalComponentExecutionData


@pipeline_component(id="fft interpolation", type=InterpolationCore, meta_info=fft_interpolation_core_meta_info)
class FFTInterpolationCore(InterpolationCore):
    """
    Computes the Fourier weights for trigonometric interpolation using FFT.
    """
    ###############################
    ### Attributes of instances ###
    ###############################
    _compiled_jax_callable_: callable



    ###################
    ### Constructor ###
    ###################
    def __init__(self, pipeline_data: list[PipelineData], additional_execution_data: AdditionalComponentExecutionData) -> None:
        super().__init__(pipeline_data, additional_execution_data)
        data: PipelineData = pipeline_data[0]

        nodes: jnp.ndarray = data.interpolation_nodes
        interpolation_values: jnp.ndarray = data.interpolation_values
        interpolation_interval: jnp.ndarray = data.interpolation_interval

        if data.data_type is not None:
            nodes = nodes.astype(data.data_type)
            interpolation_values = interpolation_values.astype(data.data_type)
            interpolation_interval = interpolation_interval.astype(data.data_type)

        self._compiled_jax_callable_ = self._create_compiled_callable_(interpolation_values)



    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        pipeline_data: PipelineData = self._pipeline_data_[0]

        weights: jnp.ndarray = self._compiled_jax_callable_()

        interpolant = FastFourierTransformationInterpolant(
            nodes=pipeline_data.interpolation_nodes,
            weights=weights,
            interval=pipeline_data.interpolation_interval
        )

        pipeline_data.interpolant = interpolant
        return pipeline_data



    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _create_compiled_callable_(interpolation_values: jnp.ndarray) -> callable:

        def _internal_perform_action_() -> jnp.ndarray:
            # Get the weights
            return jnp.fft.fft(interpolation_values) / interpolation_values.size

        return (
            jax.jit(_internal_perform_action_)       # → XLA-compatible HLO
                .lower()    # → Low-Level-IR
                .compile()  # → executable Binary
        )