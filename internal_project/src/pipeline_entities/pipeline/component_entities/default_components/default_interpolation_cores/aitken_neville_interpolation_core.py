import jax
import jax.numpy as jnp
from jax import block_until_ready
from jax.typing import DTypeLike

from functions.defaults.default_interpolants.aitken_neville_interpolant import AitkenNevilleInterpolant
from data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.interpolation_cores.aitken_neville_interpolation_core_meta_info import aitken_neville_interpolation_core_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData


@pipeline_component(id="aitken neville interpolation", type=InterpolationCore, meta_info=aitken_neville_interpolation_core_meta_info)
class AitkenNevilleInterpolationCore(InterpolationCore):
    """
    TODO
    """


    ###############################
    ### Attributes of instances ###
    ###############################
    _compiled_jax_callable_: callable


    ###################
    ### Constructor ###
    ###################
    def __init__(self, pipeline_data: list[PipelineData], additional_execution_data: AdditionalComponentExecutionData) -> None:
        super().__init__(pipeline_data, additional_execution_data)

        data_type: DTypeLike = pipeline_data[0].data_type

        nodes_dummy: jnp.ndarray = jnp.empty_like(pipeline_data[0].interpolation_nodes, dtype=data_type)
        values_dummy: jnp.ndarray = jnp.empty_like(pipeline_data[0].interpolation_values, dtype=data_type)

        self._compiled_jax_callable_ = (
            jax.jit(self._internal_perform_action_)
            .lower(nodes_dummy, values_dummy)
            .compile()
        )


    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        pipeline_data: PipelineData = self._pipeline_data_[0]

        data_type: DTypeLike = pipeline_data.data_type
        nodes = pipeline_data.interpolation_nodes.astype(data_type)
        values = pipeline_data.interpolation_values.astype(data_type)

        coefficients: jnp.ndarray = self._compiled_jax_callable_(nodes, values)
        block_until_ready(coefficients)

        interpolant = AitkenNevilleInterpolant("Aitken-Neville", coefficients)

        pipeline_data.interpolant = interpolant
        return pipeline_data


    @staticmethod
    def _internal_perform_action_(nodes: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
        n: int = len(nodes)

        initial_polynomials = jnp.zeros((n, n)).at[:, 0].set(values)

        def outer_loop(k, polynomials_outer):
            def inner_loop(i, polynomials_inner):
                node_difference: jnp.ndarray = nodes[i] - nodes[i - k]
                polynomial_difference: jnp.ndarray = (polynomials_outer[i] - polynomials_outer[i - 1])

                summand1: jnp.ndarray = polynomials_outer[i]
                summand2: jnp.ndarray = (-nodes[i] / node_difference * polynomial_difference)
                summand3: jnp.ndarray = jnp.roll(polynomial_difference / node_difference, 1, axis=0)

                return polynomials_inner.at[i].set(summand1 + summand2 + summand3)

            return jax.lax.fori_loop(k, n, inner_loop, polynomials_outer)

        return jax.lax.fori_loop(1, n, outer_loop, initial_polynomials)[n - 1]
