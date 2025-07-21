import jax.numpy as jnp

from jax.typing import DTypeLike

from interpolants.abstracts.compilable_interpolant import CompilableInterpolant
from interpolants.abstracts.compiled_interpolant import CompiledInterpolant
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.evaluation_components.interpolant_evaluator_meta_info import \
    interpolant_evaluator_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.evaluator_component import EvaluatorComponent
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import \
    pipeline_component
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import \
    AdditionalComponentExecutionData


@pipeline_component(id="interpolant evaluator", type=EvaluatorComponent, meta_info=interpolant_evaluator_meta_info)
class InterpolantEvaluator(EvaluatorComponent):
    ###############################
    ### Attributes of instances ###
    ###############################
    _interpolant_evaluation_points_: jnp.ndarray



    ###################
    ### Constructor ###
    ###################
    def __init__(self, pipeline_data: list[PipelineData], additional_execution_info: AdditionalComponentExecutionData) -> None:
        super().__init__(pipeline_data, additional_execution_info)

        data: PipelineData = pipeline_data[0]
        data_type: DTypeLike = data.data_type
        self._interpolant_evaluation_points_ = data.interpolant_evaluation_points.astype(data_type)



    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        data: PipelineData = self._pipeline_data_[0]
        overridden_attributes: dict[str, object] = self._additional_execution_info_.overridden_attributes

        interpolant: CompilableInterpolant = data.interpolant
        compiled_interpolant: CompiledInterpolant = interpolant.compile(len(self._interpolant_evaluation_points_), data.data_type, **overridden_attributes)

        data.interpolant_values = compiled_interpolant.evaluate(self._interpolant_evaluation_points_)
        return data



    ##########################
    ### Overridden methods ###
    ##########################
    def __repr__(self):
        return f"{self.__class__.__name__}"

    def __str__(self):
        return self.__repr__()



    # TODO
    # def __hash__(self):
    #     pass



    # TODO
    # def __eq__(self, other):
    #     return isinstance(other, self.__class__)



