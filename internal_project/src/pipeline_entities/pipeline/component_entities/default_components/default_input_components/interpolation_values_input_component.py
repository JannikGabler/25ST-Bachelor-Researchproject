import jax.numpy as jnp

from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.input_components.interpolation_values_input_component_meta_info import \
    interpolation_values_input_component_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.input_pipeline_component import InputPipelineComponent
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.large_data_classes.pipeline_input.pipeline_input import PipelineInput


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