from functions.abstracts.compilable_function import CompilableFunction
from functions.defaults.piecewise_sympy_expression_function import (
    PiecewiseSympyExpressionFunction,
)
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.input_components.piecewise_function_expression_input_component_meta_info import (
    piecewise_function_expression_input_component_meta_info,
)
from pipeline_entities.pipeline.component_entities.default_component_types.input_pipeline_component import (
    InputPipelineComponent,
)
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import (
    pipeline_component,
)
from data_classes.pipeline_data.pipeline_data import PipelineData
from data_classes.pipeline_input.pipeline_input import PipelineInput


@pipeline_component(
    id="piecewise function expression input",
    type=InputPipelineComponent,
    meta_info=piecewise_function_expression_input_component_meta_info,
)
class PiecewiseFunctionExpressionInputComponent(InputPipelineComponent):

    ##########################
    ### Overridden methods ###
    ##########################
    def perform_action(self) -> PipelineData:
        pipeline_data: PipelineData = self._pipeline_data_[0]

        pipeline_input: PipelineInput = self._additional_execution_info_.pipeline_input
        function_expressions: list[tuple[tuple[float, float], str]] = (
            pipeline_input.piecewise_function_expression
        )
        simplify_expression: bool = (
            pipeline_input.sympy_function_expression_simplification
        )

        compilable_function: CompilableFunction = PiecewiseSympyExpressionFunction(
            "Original function", function_expressions, simplify_expression
        )
        pipeline_data.original_function = compilable_function

        return pipeline_data
