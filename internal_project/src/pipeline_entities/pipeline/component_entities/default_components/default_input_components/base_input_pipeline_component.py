from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.input_components.base_input_pipeline_component_meta_info import base_input_pipeline_component_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.input_pipeline_component import InputPipelineComponent
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from data_classes.pipeline_data.pipeline_data import PipelineData
from data_classes.pipeline_input.pipeline_input import PipelineInput


@pipeline_component(id="base input", type=InputPipelineComponent, meta_info=base_input_pipeline_component_meta_info)
class BaseInputPipelineComponent(InputPipelineComponent):
    """
    Pipeline component that initializes the base input data for interpolation.
    It transfers information from the pipeline input into the pipeline data object.
    """

    def perform_action(self) -> PipelineData:
        """
        Transfer pipeline input values into the pipeline data object.

        Returns:
            PipelineData: Updated pipeline data populated with input values.
        """

        pipeline_data: PipelineData = self._pipeline_data_[0]
        pipeline_input: PipelineInput = self._additional_execution_info_.pipeline_input

        pipeline_data.data_type = pipeline_input.data_type
        pipeline_data.node_count = pipeline_input.node_count
        pipeline_data.interpolation_interval = pipeline_input.interpolation_interval
        pipeline_data.interpolant_evaluation_points = pipeline_input.interpolant_evaluation_points

        for key, value in pipeline_input.additional_directly_injected_values.items():
            pipeline_data.additional_values[key] = value

        return pipeline_data
