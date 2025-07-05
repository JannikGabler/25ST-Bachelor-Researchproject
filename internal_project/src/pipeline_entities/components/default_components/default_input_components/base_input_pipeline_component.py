from pipeline_entities.component_meta_info.default_component_meta_infos.input_components.base_input_pipeline_component_meta_info import \
    base_input_pipeline_component_meta_info
from pipeline_entities.components.abstracts.input_pipeline_component import InputPipelineComponent
from pipeline_entities.components.decorators.pipeline_component import pipeline_component
from pipeline_entities.data_transfer.pipeline_data import PipelineData
from pipeline_entities.pipeline_input.pipeline_input import PipelineInput


@pipeline_component(id="Base Input", type=InputPipelineComponent, meta_info=base_input_pipeline_component_meta_info)
class BaseInputPipelineComponent(InputPipelineComponent):

    def perform_action(self) -> PipelineData:
        pipeline_data: PipelineData = self._pipeline_data_[0]
        pipeline_input: PipelineInput = self._additional_execution_info_.pipeline_input

        pipeline_data.data_type = pipeline_input.data_type
        pipeline_data.node_count = pipeline_input.node_count
        pipeline_data.interpolation_interval = pipeline_input.interpolation_interval

        for key, value in pipeline_input.additional_directly_injected_values.items():
            pipeline_data.additional_values[key] = value

        return pipeline_data


