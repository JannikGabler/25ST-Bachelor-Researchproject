import jax
import jax.numpy as jnp

from pipeline_entities.component_meta_info.default_component_meta_infos.input_components.function_callable_input_component_meta_info import \
    function_callable_input_component_meta_info
from pipeline_entities.components.abstracts.input_pipeline_component import InputPipelineComponent
from pipeline_entities.components.decorators.pipeline_component import pipeline_component
from pipeline_entities.data_transfer.pipeline_data import PipelineData
from pipeline_entities.pipeline_input.pipeline_input import PipelineInput


@pipeline_component(id="FunctionCallableInput", type=InputPipelineComponent, meta_info=function_callable_input_component_meta_info)
class FunctionCallableInputComponent(InputPipelineComponent):
    ###############################
    ### Attributes of instances ###
    ###############################
    _compiled_jax_callable_: callable



    ###################
    ### Constructor ###
    ###################
    def __init__(self, pipeline_input: PipelineInput, pipeline_data: PipelineData):
        super().__init__(pipeline_input, pipeline_data)
        self._compiled_jax_callable_ = self._compile_jax_callable_()



    ##########################
    ### Overridden methods ###
    ##########################
    def perform_action(self) -> None:
        nodes: jnp.ndarray = self._pipeline_data_.nodes

        function_values: jnp.ndarray = self._compiled_jax_callable_(nodes)

        self._pipeline_data_.function_values = function_values



    #######################
    ### Private methods ###
    #######################
    def _compile_jax_callable_(self) -> callable:
        function_callable: callable = self._pipeline_input_.function_callable
        node_count: int = self._pipeline_data_.node_count
        data_type: type = self._pipeline_data_.data_type

        dummy_argument = jnp.empty(node_count, dtype=data_type)

        # Ahead-of-time compilation
        return jax.jit(function_callable).lower(dummy_argument).compile()
