import jax

from abc import ABC, abstractmethod

from data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import \
    AdditionalComponentExecutionData


class AOTCInterpolationCore(InterpolationCore, ABC):
    _compiled_jax_callable_: callable



    def __init__(self, pipeline_data: list[PipelineData], additional_execution_data: AdditionalComponentExecutionData) -> None:
        super().__init__(pipeline_data, additional_execution_data)

        internal_perform_action_function = self._get_internal_perform_action_function_()
        self._compiled_jax_callable_ = jax.jit(internal_perform_action_function).lower().compile()



    @abstractmethod
    def _get_internal_perform_action_function_(self) -> callable:
        pass