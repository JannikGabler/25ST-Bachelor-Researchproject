import jax
import jax.numpy as jnp

from interpolants.default_interpolants.aitken_neville_interpolant import AitkenNevilleInterpolant
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.interpolation_cores.aitken_neville_interpolation_core_meta_info import \
    aitken_neville_interpolation_core_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import \
    pipeline_component
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import \
    AdditionalComponentExecutionData


@pipeline_component(id="aitken neville interpolation", type=InterpolationCore, meta_info=aitken_neville_interpolation_core_meta_info)
class AitkenNevilleInterpolationCore(InterpolationCore):
    """
    TODO
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

        if data.data_type is not None:
            data.interpolation_nodes = data.interpolation_nodes.astype(data.data_type)
            data.interpolation_values = data.interpolation_values.astype(data.data_type)



    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        pipeline_data: PipelineData = self._pipeline_data_[0]

        interpolant = AitkenNevilleInterpolant(
            nodes=pipeline_data.interpolation_nodes,
            values=pipeline_data.interpolation_values,
        )

        pipeline_data.interpolant = interpolant

        return pipeline_data



    @staticmethod
    def _create_compiled_callable_(nodes: jnp.ndarray, interpolation_values: jnp.ndarray, data_type: jnp.dtype) -> callable:
        def _internal_perform_action_() -> jnp.ndarray:
            # Number of interpolation points
            n = nodes.size - 1

            initial_coefficients: jnp.ndarray = interpolation_values.copy()#((n, n), dtype=data_type)

            def outer_loop(k: int, polynomials: jnp.ndarray) -> jnp.ndarray:
                new_polynomials: jnp.ndarray = jnp.empty((n + 1 - k, k + 1), dtype=data_type)

                def inner_loop(i: int, polynomials) -> jnp.ndarray:
                    new_polynomials[i] = polynomials[k - i]
                    difference = polynomials[i] - polynomials[i - 1]
                    new_polynomials[i] -= difference * nodes[i] / (nodes[i] - nodes[i - k])
                    new_polynomials[i] += jnp.pad((difference / (nodes[i] - nodes[i - k])), (1, 0))

            jax.lax.fori_loop(1, n + 1, outer_loop, initial_coefficients)


            return coefficients

        return (
            jax.jit(_internal_perform_action_)  # → XLA-compatible HLO
            .lower()  # → Low-Level-IR
            .compile()  # → executable Binary
        )

