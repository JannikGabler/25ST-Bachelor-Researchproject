import jax
import jax.numpy as jnp
import sympy
from sympy import Expr

from pipeline_entities.component_meta_info.default_component_meta_infos.input_components.single_function_expression_input_component_meta_info import \
    single_function_expression_input_component_meta_info
from pipeline_entities.components.abstracts.input_pipeline_component import InputPipelineComponent
from pipeline_entities.components.decorators.pipeline_component import pipeline_component
from pipeline_entities.data_transfer.pipeline_data import PipelineData
from pipeline_entities.pipeline_input.pipeline_input import PipelineInput


@pipeline_component(id="SingleFunctionExpressionInput", type=InputPipelineComponent, meta_info=single_function_expression_input_component_meta_info)
class SingleFunctionExpressionInputComponent(InputPipelineComponent):
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
        function_expression: str = self._pipeline_input_.single_function_expression
        simplify_expression: bool = self._pipeline_input_.sympy_function_expression_simplification

        x = sympy.symbols("x")
        expr: Expr = sympy.sympify(function_expression, evaluate=simplify_expression)

        scalar_lambda = sympy.lambdify(x, expr, modules="jax")

        # Vectorize scalar lambda
        return jax.vmap(scalar_lambda)