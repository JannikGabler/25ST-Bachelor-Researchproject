from typing import Any

import jax
import sympy
from sympy import Expr

from pipeline_entities.pipeline.component_entities.component_meta_info.default_component_meta_infos.input_components.piecewise_function_expression_input_component_meta_info import \
    piecewise_function_expression_input_component_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.input_pipeline_component import InputPipelineComponent
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.large_data_classes.pipeline_input.pipeline_input import PipelineInput


@pipeline_component(id="piecewise function expression input", type=InputPipelineComponent, meta_info=piecewise_function_expression_input_component_meta_info)
class PiecewiseFunctionExpressionInputComponent(InputPipelineComponent):

    ##########################
    ### Overridden methods ###
    ##########################
    def perform_action(self) -> PipelineData:
        pipeline_data: PipelineData = self._pipeline_data_[0]

        function_callable: callable = self._create_jax_lambda_()

        pipeline_data.function_callable = function_callable
        return pipeline_data



    #######################
    ### Private methods ###
    #######################
    def _create_jax_lambda_(self) -> callable:
        x = sympy.symbols("x")
        piecewise_sympy_expression: Expr = self._create_piecewise_sympy_expression(x)

        lambda_scalar = sympy.lambdify(x, piecewise_sympy_expression, modules="jax")

        return jax.vmap(lambda_scalar)



    def _create_piecewise_sympy_expression(self, x: Any) -> Expr:
        pipeline_input: PipelineInput = self._additional_execution_info_.pipeline_input

        function_expressions: list[tuple[tuple[float, float], str]] = pipeline_input.piecewise_function_expression
        simplify_expression: bool = pipeline_input.sympy_function_expression_simplification

        expressions: list[tuple[Expr, any]] = []

        for (lower, upper), function_expression in function_expressions:
            expression: Expr = sympy.sympify(function_expression, evaluate=simplify_expression)
            condition = (x >= lower) & (x < upper)
            expressions.append((expression, condition))

        return sympy.Piecewise(*expressions)

