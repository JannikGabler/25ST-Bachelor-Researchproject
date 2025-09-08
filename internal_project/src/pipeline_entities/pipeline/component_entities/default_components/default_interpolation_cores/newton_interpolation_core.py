import jax
from jax import block_until_ready

from jax.typing import DTypeLike

from functions.defaults.default_interpolants.newton_interpolant import NewtonInterpolant
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.interpolation_cores.newton_interpolation_core_meta_info import \
    newton_interpolation_core_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.aotc_interpolation_core import \
    AOTCInterpolationCore
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore
import jax.numpy as jnp

from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData


@pipeline_component(id="newton interpolation", type=InterpolationCore, meta_info=newton_interpolation_core_meta_info)
class NewtonInterpolationCore(InterpolationCore):
    """
    Computes the coefficients of the Newton interpolation polynomial using divided differences.

    Returns:
         coefficients: 1D array of coefficients computed via divided differences.
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

        self._compiled_jax_callable_ = jax.jit(self._internal_perform_action_).lower(nodes_dummy, values_dummy).compile()



    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        pd: PipelineData = self._pipeline_data_[0]

        nodes: jnp.ndarray = pd.interpolation_nodes.astype(pd.data_type)
        values: jnp.ndarray = pd.interpolation_values.astype(pd.data_type)

        divided_differences: jnp.ndarray = self._compiled_jax_callable_(nodes, values)
        block_until_ready(divided_differences)

        interpolant = NewtonInterpolant(
            name="Newton",
            nodes=pd.interpolation_nodes,
            divided_differences=divided_differences
        )

        pd.interpolant = interpolant
        return pd



    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _internal_perform_action_(nodes: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
        n: int = nodes.shape[0]

        # Initialize coefficients array with function values
        coefficients = values.copy()

        # Compute divided differences of increasing order
        def outer_loop(j, coef_outer):
            # Update entries for current order of divided differences
            def inner_loop(i, coef_inner):
                numerator = coef_outer[i] - coef_outer[i - 1]
                denominator = nodes[i] - nodes[i - j]
                return coef_inner.at[i].set(numerator / denominator)

            # Apply the inner loop to update coefficients for the current order
            coef_outer = jax.lax.fori_loop(j, n, inner_loop, coef_outer)
            return coef_outer

        # Apply the outer loop to compute all orders of divided differences
        coefficients = jax.lax.fori_loop(1, n, outer_loop, coefficients)

        return coefficients


    # def _internal_perform_action_(self) -> jnp.ndarray:
    #     data_type: DTypeLike = self._pipeline_data_[0].data_type
    #     nodes = self._pipeline_data_[0].interpolation_nodes.astype(data_type)
    #     values = self._pipeline_data_[0].interpolation_values.astype(data_type)
    #     n = nodes.size
    #
    #     initial_divided_differences: jnp.ndarray = values.copy()
    #
    #     def calc_node_differences(k: int) -> jnp.ndarray:
    #         length: int = n - k
    #         end_nodes: jnp.ndarray = jax.lax.dynamic_slice(nodes, (k, ), (length, )) #nodes[k:]
    #         start_nodes: jnp.ndarray = jax.lax.dynamic_slice(nodes, (0, ), (length, ))  #nodes[:-k]
    #         return end_nodes - start_nodes
    #
    #     def calc_divided_difference_differences(k: int, divided_differences: jnp.ndarray) -> jnp.ndarray:
    #         length: int = n - k
    #         first_divided_differences: jnp.ndarray = jax.lax.dynamic_slice(divided_differences, (k, ), (length, )) #divided_differences[k:]
    #         second_divided_differences: jnp.ndarray = jax.lax.dynamic_slice(divided_differences, (k-1, ), (length, )) #divided_differences[k-1:-1]
    #         return first_divided_differences - second_divided_differences
    #
    #     def loop(k: int, old_divided_differences: jnp.ndarray) -> jnp.ndarray:
    #         node_differences: jnp.ndarray = calc_node_differences(k)
    #         divided_difference_differences: jnp.ndarray = calc_divided_difference_differences(k, old_divided_differences)
    #         return old_divided_differences.at[k:].set(divided_difference_differences / node_differences)
    #
    #     return jax.lax.fori_loop(1, n, loop, initial_divided_differences)



    # def _internal_perform_action_(self) -> jnp.ndarray:
    #     data_type: DTypeLike = self._pipeline_data_[0].data_type
    #     nodes = self._pipeline_data_[0].interpolation_nodes.astype(data_type)
    #     values = self._pipeline_data_[0].interpolation_values.astype(data_type)
    #     n = nodes.size
    #
    #     divided_differences: jnp.ndarray = values.copy()
    #
    #     def outer_loop(j, coef_outer):
    #
    #         def inner_loop(i, coef_inner):
    #             numerator = coef_outer[i] - coef_outer[i - 1]
    #             denominator = nodes[i] - nodes[i - j]
    #             return coef_inner.at[i].set(numerator / denominator)
    #
    #         # Apply the inner loop to update coefficients for the current order
    #         coef_outer = jax.lax.fori_loop(j, n, inner_loop, coef_outer)
    #         return coef_outer
    #
    #     # Apply the outer loop to compute all orders of divided differences
    #     coefficients = jax.lax.fori_loop(1, n, outer_loop, coefficients)
    #
    #     return coefficients