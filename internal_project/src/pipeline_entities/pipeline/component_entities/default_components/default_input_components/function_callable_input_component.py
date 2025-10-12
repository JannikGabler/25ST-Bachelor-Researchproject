from data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.input_components.function_callable_input_component_meta_info import (
    function_callable_input_component_meta_info,
)
from pipeline_entities.pipeline.component_entities.default_component_types.input_pipeline_component import (
    InputPipelineComponent,
)
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import (
    pipeline_component,
)


@pipeline_component(
    id="function callable input",
    type=InputPipelineComponent,
    meta_info=function_callable_input_component_meta_info,
)
class FunctionCallableInputComponent(InputPipelineComponent):

    ##########################
    ### Overridden methods ###
    ##########################
    def perform_action(self) -> PipelineData:
        pipeline_data: PipelineData = self._pipeline_data_[0]
        function_callable: callable = (
            self._additional_execution_info_.pipeline_input.function_callable
        )

        pipeline_data.original_function = function_callable
        return pipeline_data
