import jax

from abc import ABC, abstractmethod

from data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData


class AOTCInterpolationCore(InterpolationCore, ABC):
    """
    Abstract base class for interpolation cores using AOTC (ahead-of-time compilation).
    It compiles an internal JAX function once during initialization and provides the compiled callable to derived components.
    """

    _compiled_jax_callable_: callable


    def __init__(self, pipeline_data: list[PipelineData], additional_execution_data: AdditionalComponentExecutionData) -> None:
        """
        Initialize the AOTC interpolation core and compile its internal JAX function.

        Args:
            pipeline_data (list[PipelineData]): Pipeline data required by the interpolation core.
            additional_execution_data (AdditionalComponentExecutionData): Additional execution data.
        """

        super().__init__(pipeline_data, additional_execution_data)
        internal_perform_action_function = self._get_internal_perform_action_function_()
        self._compiled_jax_callable_ = jax.jit(internal_perform_action_function).lower().compile()


    @abstractmethod
    def _get_internal_perform_action_function_(self) -> callable:
        pass
