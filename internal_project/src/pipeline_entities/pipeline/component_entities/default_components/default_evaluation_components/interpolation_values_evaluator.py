import jax.numpy as jnp

from jax.typing import DTypeLike

from functions.abstracts.compilable_function import CompilableFunction
from functions.abstracts.compiled_function import CompiledFunction
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.evaluation_components.interpolation_values_evaluator_meta_info import (
    interpolation_values_evaluator_meta_info,
)
from pipeline_entities.pipeline.component_entities.default_component_types.evaluator_component import (
    EvaluatorComponent,
)
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import (
    pipeline_component,
)
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import (
    AdditionalComponentExecutionData,
)
from data_classes.pipeline_data.pipeline_data import PipelineData


@pipeline_component(
    id="interpolation values evaluator",
    type=EvaluatorComponent,
    meta_info=interpolation_values_evaluator_meta_info,
)
class InterpolationValuesEvaluator(EvaluatorComponent):
    ###############################
    ### Attributes of instances ###
    ###############################
    _compiled_original_function_: CompiledFunction

    ###################
    ### Constructor ###
    ###################
    def __init__(
        self,
        pipeline_data: list[PipelineData],
        additional_execution_info: AdditionalComponentExecutionData,
    ) -> None:
        super().__init__(pipeline_data, additional_execution_info)

        amount_of_evaluation_points: int = self._pipeline_data_[0].node_count
        data_type: DTypeLike = self._pipeline_data_[0].data_type
        overridden_attributes: dict[str, object] = (
            self._additional_execution_info_.overridden_attributes
        )

        compilable_original_function: CompilableFunction = self._pipeline_data_[
            0
        ].original_function
        self._compiled_original_function_ = compilable_original_function.compile(
            amount_of_evaluation_points, data_type, **overridden_attributes
        )

        # self._compiled_jax_callable_ = jax.jit(self._internal_perform_action_).lower().compile()

        # nodes: jnp.ndarray = self._pipeline_data_[0].interpolation_nodes
        # function_callable: Callable[[jnp.ndarray], jnp.ndarray] = self._pipeline_data_[0].function_callable
        # specified_data_type: type = self._pipeline_data_[0].data_type
        #
        # if specified_data_type == nodes.dtype:
        #     self._compiled_jax_callable_ = self._create_compiled_callable_(nodes, function_callable)
        # else:
        #     self._compiled_jax_callable_ = self._create_data_type_converting_compiled_callable_(nodes, function_callable, specified_data_type)

    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        data: PipelineData = self._pipeline_data_[0]
        interpolation_nodes: jnp.ndarray = data.interpolation_nodes.astype(
            data.data_type
        )

        interpolation_values: jnp.ndarray = self._compiled_original_function_.evaluate(
            interpolation_nodes
        )

        data.interpolation_values = interpolation_values
        return data

    ##########################
    ### Overridden methods ###
    ##########################
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return f"{self.__class__.__name__}()"

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __hash__(self):
        return hash(self.__class__.__name__)

    #######################
    ### Private methods ###
    #######################
    # def _internal_perform_action_(self) -> jnp.ndarray:
    #     data_type: DTypeLike = self._pipeline_data_[0].data_type
    #     function_callable: callable = self._pipeline_data_[0].original_function
    #     interpolation_nodes: jnp.ndarray = self._pipeline_data_[0].interpolation_nodes
    #
    #     return function_callable(interpolation_nodes.astype(data_type))

    # @staticmethod
    # def _create_compiled_callable_(interpolation_nodes: jnp.ndarray, function_callable: Callable[[jnp.ndarray], jnp.ndarray]) -> callable:
    #
    #     def _internal_perform_action_() -> jnp.ndarray:
    #         return function_callable(interpolation_nodes)
    #
    #     return (
    #         jax.jit(_internal_perform_action_)  # → XLA-compatible HLO
    #         .lower()  # → Low-Level-IR
    #         .compile()  # → executable Binary
    #     )

    # @staticmethod
    # def _create_data_type_converting_compiled_callable_(interpolation_nodes: jnp.ndarray, function_callable: Callable[[jnp.ndarray], jnp.ndarray],
    #         new_data_type: type) -> callable:
    #
    #
    #
    #     return (
    #         jax.jit(_internal_perform_action_)  # → XLA-compatible HLO
    #         .lower()  # → Low-Level-IR
    #         .compile()  # → executable Binary
    #     )
