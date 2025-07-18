from typing import Callable

import jax
import jax.numpy as jnp

from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.evaluation_components.interpolation_values_evaluator_meta_info import \
    interpolation_values_evaluator_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.evaluator_component import EvaluatorComponent
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData


@pipeline_component(id="Interpolation Values Evaluator", type=EvaluatorComponent, meta_info=interpolation_values_evaluator_meta_info)
class InterpolationValuesEvaluator(EvaluatorComponent):
    ###############################
    ### Attributes of instances ###
    ###############################
    _compiled_jax_callable_: callable



    ###################
    ### Constructor ###
    ###################
    def __init__(self, pipeline_data: list[PipelineData], additional_execution_info: AdditionalComponentExecutionData) -> None:
        super().__init__(pipeline_data, additional_execution_info)
        nodes: jnp.ndarray = self._pipeline_data_[0].interpolation_nodes
        function_callable: Callable[[jnp.ndarray], jnp.ndarray] = self._pipeline_data_[0].function_callable
        specified_data_type: type = self._pipeline_data_[0].data_type

        if specified_data_type == nodes.dtype:
            self._compiled_jax_callable_ = self._create_compiled_callable_(nodes, function_callable)
        else:
            self._compiled_jax_callable_ = self._create_data_type_converting_compiled_callable_(nodes, function_callable, specified_data_type)



    ##########################
    ### Overridden methods ###
    ##########################
    def perform_action(self) -> PipelineData:
        pipeline_data = self._pipeline_data_[0]

        interpolation_values = self._compiled_jax_callable_()

        pipeline_data.interpolation_values = interpolation_values
        return pipeline_data



    @staticmethod
    def _create_compiled_callable_(interpolation_nodes: jnp.ndarray, function_callable: Callable[[jnp.ndarray], jnp.ndarray]) -> callable:

        def _internal_perform_action_() -> jnp.ndarray:
            return function_callable(interpolation_nodes)

        return (
            jax.jit(_internal_perform_action_)  # → XLA-compatible HLO
            .lower()  # → Low-Level-IR
            .compile()  # → executable Binary
        )



    @staticmethod
    def _create_data_type_converting_compiled_callable_(interpolation_nodes: jnp.ndarray, function_callable: Callable[[jnp.ndarray], jnp.ndarray],
            new_data_type: type) -> callable:

        def _internal_perform_action_() -> jnp.ndarray:
            return function_callable(interpolation_nodes.astype(new_data_type))

        return (
            jax.jit(_internal_perform_action_)  # → XLA-compatible HLO
            .lower()  # → Low-Level-IR
            .compile()  # → executable Binary
        )
