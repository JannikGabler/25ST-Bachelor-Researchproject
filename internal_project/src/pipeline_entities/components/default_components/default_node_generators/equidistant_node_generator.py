import jax

from pipeline_entities.component_meta_info.default_component_meta_infos.node_generators.equidistant_node_generator_meta_info import \
    equidistant_node_generator_meta_info
from pipeline_entities.components.abstracts.node_generator import NodeGenerator
import jax.numpy as jnp

from pipeline_entities.components.decorators.pipeline_component import pipeline_component
from pipeline_entities.data_transfer.pipeline_data import PipelineData


@pipeline_component(id="equidistant node generator", type=NodeGenerator, meta_info=equidistant_node_generator_meta_info)
class EquidistantNodeGenerator(NodeGenerator):
    ###############################
    ### Attributes of instances ###
    ###############################
    _compiled_jax_callable_: callable



    ###################
    ### Constructor ###
    ###################
    def __init__(self, pipeline_data: PipelineData) -> None:
        super().__init__(pipeline_data)

        data_type: type = pipeline_data.data_type
        node_count: int = pipeline_data.node_count
        interpolation_interval: jnp.ndarray = pipeline_data.interpolation_interval

        self._compiled_jax_callable_ = self._create_compiled_callable_(data_type, node_count, interpolation_interval)




    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> None:
        nodes = self._compiled_jax_callable_()
        self._pipeline_data_.nodes = nodes



    #######################
    ### Private methods ###
    #######################
    def _create_compiled_callable_(self, data_type: type, node_count: int, interpolation_interval: jnp.ndarray):

        def _internal_perform_action_() -> jnp.ndarray:
            return jnp.linspace(interpolation_interval[0], interpolation_interval[1], node_count,
                                dtype=data_type)

        return (
            jax.jit(_internal_perform_action_)       # → XLA-compatible HLO
                .lower()    # → Low-Level-IR
                .compile()  # → executable Binary
        )