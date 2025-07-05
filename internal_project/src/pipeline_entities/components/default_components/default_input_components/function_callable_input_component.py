import jax
import jax.numpy as jnp

from pipeline_entities.component_meta_info.default_component_meta_infos.input_components.function_callable_input_component_meta_info import \
    function_callable_input_component_meta_info
from pipeline_entities.components.abstracts.input_pipeline_component import InputPipelineComponent
from pipeline_entities.components.decorators.pipeline_component import pipeline_component
from pipeline_entities.data_transfer.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.data_transfer.pipeline_data import PipelineData
from pipeline_entities.pipeline_input.pipeline_input import PipelineInput


@pipeline_component(id="function callable input", type=InputPipelineComponent, meta_info=function_callable_input_component_meta_info)
class FunctionCallableInputComponent(InputPipelineComponent):

    ##########################
    ### Overridden methods ###
    ##########################
    def perform_action(self) -> PipelineData:
        pipeline_data: PipelineData = self._pipeline_data_[0]
        function_callable: callable = self._additional_execution_info_.pipeline_input.function_callable

        pipeline_data.function_callable = function_callable
        return pipeline_data


