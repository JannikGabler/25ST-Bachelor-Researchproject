import jax.numpy as jnp

from pipeline_entities.component_meta_info.default_component_meta_infos.input_components.interpolation_values_input_component_meta_info import \
    interpolation_values_input_component_meta_info
from pipeline_entities.components.abstracts.input_pipeline_component import InputPipelineComponent
from pipeline_entities.components.decorators.pipeline_component import pipeline_component
from pipeline_entities.data_transfer.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.data_transfer.pipeline_data import PipelineData
from pipeline_entities.pipeline_input.pipeline_input import PipelineInput


@pipeline_component(id="interpolation values input", type=InputPipelineComponent, meta_info=interpolation_values_input_component_meta_info)
class InterpolationValuesInputComponent(InputPipelineComponent):
    ###################
    ### Constructor ###
    ###################
    def __init__(self, pipeline_data: list[PipelineData], additional_execution_data: AdditionalComponentExecutionData) -> None:
        super().__init__(pipeline_data, additional_execution_data)



    ##########################
    ### Overridden methods ###
    ##########################
    def perform_action(self) -> PipelineData:
        pipeline_data: PipelineData = self._pipeline_data_[0]
        pipeline_input: PipelineInput = self._additional_execution_info_.pipeline_input

        interpolation_values: jnp.ndarray = pipeline_input.interpolation_values

        pipeline_data.interpolation_nodes = interpolation_values
        return pipeline_data