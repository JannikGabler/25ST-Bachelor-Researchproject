import jax
import jax.numpy as jnp
import sympy
from sympy import Expr

from pipeline_entities.component_meta_info.default_component_meta_infos.input_components.function_expression_input_component_meta_info import \
    function_expression_input_component_meta_info
from pipeline_entities.components.abstracts.input_pipeline_component import InputPipelineComponent
from pipeline_entities.components.decorators.pipeline_component import pipeline_component
from pipeline_entities.data_transfer.pipeline_data import PipelineData
from pipeline_entities.pipeline_input.pipeline_input import PipelineInput


@pipeline_component(id="function expression input", type=InputPipelineComponent, meta_info=function_expression_input_component_meta_info)
class FunctionExpressionInputComponent(InputPipelineComponent):

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
        pipeline_input: PipelineInput = self._additional_execution_info_.pipeline_input

        function_expression: str = pipeline_input.function_expression
        simplify_expression: bool = pipeline_input.sympy_function_expression_simplification

        x = sympy.symbols("x")
        expr: Expr = sympy.sympify(function_expression, evaluate=simplify_expression)

        scalar_lambda = sympy.lambdify(x, expr, modules="jax")

        # Vectorize scalar lambda
        return jax.vmap(scalar_lambda)