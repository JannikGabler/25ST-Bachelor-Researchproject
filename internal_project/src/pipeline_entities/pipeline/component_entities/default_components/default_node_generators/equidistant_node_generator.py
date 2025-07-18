import jax

from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.node_generators.equidistant_node_generator_meta_info import \
    equidistant_node_generator_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.node_generator import NodeGenerator
import jax.numpy as jnp

from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData


@pipeline_component(id="equidistant node generator", type=NodeGenerator, meta_info=equidistant_node_generator_meta_info)
class EquidistantNodeGenerator(NodeGenerator):
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
            return jnp.linspace(interpolation_interval[0], interpolation_interval[1], node_count,
                                dtype=data_type)

        return (
            jax.jit(_internal_perform_action_)       # → XLA-compatible HLO
                .lower()    # → Low-Level-IR
                .compile()  # → executable Binary
        )