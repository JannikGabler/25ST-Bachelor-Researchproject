import jax

from jax import numpy as jnp

from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.node_generators.second_type_chebyshev_node_generator_meta_info import \
    second_type_chebyshev_node_generator_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.node_generator import NodeGenerator
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData


@pipeline_component(id="chebyshev2 node generator", type=NodeGenerator, meta_info=second_type_chebyshev_node_generator_meta_info)
class SecondTypeChebyshevNodeGenerator(NodeGenerator):
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

        data_type: type = data.data_type
        node_count: int = data.node_count
        interpolation_interval: jnp.ndarray = data.interpolation_interval

        self._compiled_jax_callable_ = self._create_compiled_callable_(data_type, node_count, interpolation_interval)



    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        pipeline_data: PipelineData = self._pipeline_data_[0]

        nodes: jnp.ndarray = self._compiled_jax_callable_()

        pipeline_data.interpolation_nodes = nodes
        return pipeline_data



    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _create_compiled_callable_(data_type: type, node_count: int, interpolation_interval: jnp.ndarray) -> callable:

        def _internal_perform_action_() -> jnp.ndarray:
            nodes: jnp.ndarray = jnp.arange(0, node_count, dtype=data_type)
            nodes = jnp.multiply(nodes, jnp.pi / (node_count - 1))
            nodes = jnp.cos(nodes)

            do_rescale = jnp.logical_or(interpolation_interval[0] != -1, interpolation_interval[1] != 1)

            def rescale_nodes():
                old_length = 2
                new_length = interpolation_interval[1] - interpolation_interval[0]
                length_ratio = new_length / old_length

                rescaled_nodes = jnp.multiply(nodes, length_ratio)
                return jnp.add(rescaled_nodes, interpolation_interval[0] + length_ratio)

            return jax.lax.cond(do_rescale, rescale_nodes, lambda: nodes)

        return (
            jax.jit(_internal_perform_action_)  # → XLA-compatible HLO
            .lower()  # → Low-Level-IR
            .compile()  # → executable Binary
        )


