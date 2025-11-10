import jax.numpy as jnp
from jax import block_until_ready
from jax.typing import DTypeLike
from functions.abstracts.compilable_function import CompilableFunction
from functions.abstracts.compiled_function import CompiledFunction
from data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.evaluation_components.interpolant_evaluator_meta_info import interpolant_evaluator_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.evaluator_component import EvaluatorComponent
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData


@pipeline_component(id="interpolant evaluator", type=EvaluatorComponent, meta_info=interpolant_evaluator_meta_info)
class InterpolantEvaluator(EvaluatorComponent):
    """
    Pipeline component that evaluates a compiled interpolant at specified evaluation points and stores the resulting values in the pipeline data.
    """


    ###############################
    ### Attributes of instances ###
    ###############################
    _compiled_interpolant_: CompiledFunction


    ###################
    ### Constructor ###
    ###################
    def __init__(self, pipeline_data: list[PipelineData], additional_execution_info: AdditionalComponentExecutionData) -> None:
        """
        Initialize the interpolant evaluator by compiling the interpolant stored in the pipeline data.

        Args:
            pipeline_data (list[PipelineData]): Pipeline data containing the interpolant and evaluation points.
            additional_execution_info (AdditionalComponentExecutionData): Additional exeuction info.
        """

        super().__init__(pipeline_data, additional_execution_info)
        amount_of_evaluation_points: int = len(self._pipeline_data_[0].interpolant_evaluation_points)
        data_type: DTypeLike = self._pipeline_data_[0].data_type
        overridden_attributes: dict[str, object] = self._additional_execution_info_.overridden_attributes
        compilable_interpolant: CompilableFunction = self._pipeline_data_[0].interpolant
        self._compiled_interpolant_ = compilable_interpolant.compile(amount_of_evaluation_points, data_type, **overridden_attributes)


    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        """
        Evaluate the compiled interpolant and store the results in the pipeline data.

        Returns:
            PipelineData: Updated pipeline data with `interpolant_values` set.
        """

        data: PipelineData = self._pipeline_data_[0]
        interpolant_evaluation_points: jnp.ndarray = (data.interpolant_evaluation_points.astype(data.data_type))
        interpolant_values: jnp.ndarray = self._compiled_interpolant_.evaluate(interpolant_evaluation_points)
        block_until_ready(interpolant_values)
        data.interpolant_values = interpolant_values
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
