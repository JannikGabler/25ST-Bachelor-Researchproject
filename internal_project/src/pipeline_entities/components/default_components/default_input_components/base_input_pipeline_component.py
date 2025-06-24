from pipeline_entities.component_meta_info.default_component_meta_infos.input_components.base_input_pipeline_component_meta_info import \
    base_input_pipeline_component_meta_info
from pipeline_entities.components.abstracts.input_pipeline_component import InputPipelineComponent
from pipeline_entities.components.decorators.pipeline_component import pipeline_component


@pipeline_component(id="BaseInput", type=InputPipelineComponent, meta_info=base_input_pipeline_component_meta_info)
class BaseInputPipelineComponent(InputPipelineComponent):

    def perform_action(self) -> None:
        self._pipeline_data_.data_type = self._pipeline_input_.data_type
        self._pipeline_data_.node_count = self._pipeline_input_.node_count
        self._pipeline_data_.interpolation_interval = self._pipeline_input_.interpolation_interval

        for key, value in self._pipeline_input_.additional_directly_injected_values.items():
            self._pipeline_data_.additional_values[key] = value



