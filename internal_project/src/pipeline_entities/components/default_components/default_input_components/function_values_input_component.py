import jax.numpy as jnp

from pipeline_entities.component_meta_info.default_component_meta_infos.input_components.function_value_input_component_meta_info import \
    function_value_input_component_meta_info
from pipeline_entities.components.abstracts.input_pipeline_component import InputPipelineComponent
from pipeline_entities.components.decorators.pipeline_component import pipeline_component
from pipeline_entities.data_transfer.pipeline_data import PipelineData
from pipeline_entities.pipeline_input.pipeline_input import PipelineInput


@pipeline_component(id="FunctionValuesInput", type=InputPipelineComponent, meta_info=function_value_input_component_meta_info)
class FunctionValueInputComponent(InputPipelineComponent):
    ###################
    ### Constructor ###
    ###################
    def __init__(self, pipeline_input: PipelineInput, pipeline_data: PipelineData):
        super().__init__(pipeline_input, pipeline_data)



    ##########################
    ### Overridden methods ###
    ##########################
    def perform_action(self) -> None:
        function_values: jnp.ndarray = self._pipeline_input_.function_values

        self._pipeline_data_.function_values = function_values
