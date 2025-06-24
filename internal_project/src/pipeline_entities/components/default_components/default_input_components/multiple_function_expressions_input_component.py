from typing import Any

import jax
import jax.numpy as jnp
import sympy
from sympy import Expr

from pipeline_entities.component_meta_info.default_component_meta_infos.input_components.multiple_function_expressions_input_component_meta_info import \
    multiple_function_expressions_input_component_meta_info
from pipeline_entities.components.abstracts.input_pipeline_component import InputPipelineComponent
from pipeline_entities.components.decorators.pipeline_component import pipeline_component
from pipeline_entities.data_transfer.pipeline_data import PipelineData
from pipeline_entities.pipeline_input.pipeline_input import PipelineInput


@pipeline_component(id="MultipleFunctionExpressionsInput", type=InputPipelineComponent, meta_info=multiple_function_expressions_input_component_meta_info)
class MultipleFunctionExpressionInputComponent(InputPipelineComponent):
    ###############################
    ### Attributes of instances ###
    ###############################
    _compiled_jax_callable_: callable



    ###################
    ### Constructor ###
    ###################
    def __init__(self, pipeline_input: PipelineInput, pipeline_data: PipelineData):
        super().__init__(pipeline_input, pipeline_data)
        self._compiled_jax_callable_ = self._compile_jax_callable_()



    ##########################
    ### Overridden methods ###
    ##########################
    def perform_action(self) -> None:
        nodes: jnp.ndarray = self._pipeline_data_.nodes

        function_values: jnp.ndarray = self._compiled_jax_callable_(nodes)

        self._pipeline_data_.function_values = function_values



    #######################
    ### Private methods ###
    #######################
    def _compile_jax_callable_(self) -> callable:
        lambda_vectorized: callable = self._create_jax_lambda_()
        node_count: int = self._pipeline_data_.node_count
        data_type: type = self._pipeline_data_.data_type

        dummy_argument = jnp.empty(node_count, dtype=data_type)

        # Ahead-of-time compilation
        return jax.jit(lambda_vectorized).lower(dummy_argument).compile()



    def _create_jax_lambda_(self) -> callable:
        x = sympy.symbols("x")
        piecewise_sympy_expression: Expr = self._create_piecewise_sympy_expression(x)

        lambda_scalar = sympy.lambdify(x, piecewise_sympy_expression, modules="jax")

        return jax.vmap(lambda_scalar)



    def _create_piecewise_sympy_expression(self, x: Any) -> Expr:
        function_expressions: list[tuple[tuple[float, float], str]] = self._pipeline_input_.multiple_function_expressions
        simplify_expression: bool = self._pipeline_input_.sympy_function_expression_simplification

        expressions: list[tuple[Expr, any]] = []

        for (lower, upper), function_expression in function_expressions:
            expression: Expr = sympy.sympify(function_expression, evaluate=simplify_expression)
            condition = (x >= lower) & (x < upper)
            expressions.append((expression, condition))

        return sympy.Piecewise(*expressions)

