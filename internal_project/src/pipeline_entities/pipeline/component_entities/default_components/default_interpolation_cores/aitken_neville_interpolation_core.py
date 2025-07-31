import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from interpolants.default_interpolants.aitken_neville_interpolant import AitkenNevilleInterpolant
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.interpolation_cores.aitken_neville_interpolation_core_meta_info import \
    aitken_neville_interpolation_core_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.aotc_interpolation_core import \
    AOTCInterpolationCore
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import \
    pipeline_component
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import \
    AdditionalComponentExecutionData


@pipeline_component(id="aitken neville interpolation", type=InterpolationCore, meta_info=aitken_neville_interpolation_core_meta_info)
class AitkenNevilleInterpolationCore(AOTCInterpolationCore):
    """
    TODO
    """

    ###################
    ### Constructor ###
    ###################
    def __init__(self, pipeline_data: list[PipelineData], additional_execution_data: AdditionalComponentExecutionData) -> None:
        super().__init__(pipeline_data, additional_execution_data)



    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        pipeline_data: PipelineData = self._pipeline_data_[0]

        coefficients: jnp.ndarray = self._compiled_jax_callable_()
        interpolant = AitkenNevilleInterpolant("Aitken-Neville", coefficients)

        pipeline_data.interpolant = interpolant
        return pipeline_data



    ##########################
    ### Overridden methods ###
    ##########################
    def _get_internal_perform_action_function_(self) -> callable:
        return self._internal_perform_action_



    #######################
    ### Private methods ###
    #######################
    def _internal_perform_action_(self) -> jnp.ndarray:
        data_type: DTypeLike = self._pipeline_data_[0].data_type
        nodes = self._pipeline_data_[0].interpolation_nodes.astype(data_type)
        values = self._pipeline_data_[0].interpolation_values.astype(data_type)
        n: int = self._pipeline_data_[0].node_count

        # Initialize coefficients array with function values
        initial_polynomials = jnp.zeros((n, n)).at[:, 0].set(values)

        # Compute divided differences of increasing order
        def outer_loop(k, polynomials_outer):
            # Update entries for current order of divided differences
            def inner_loop(i, polynomials_inner):
                node_difference: jnp.ndarray = nodes[i] - nodes[i - k]
                polynomial_difference: jnp.ndarray = polynomials_outer[i] - polynomials_outer[i - 1]

                summand1: jnp.ndarray = polynomials_outer[i]
                summand2: jnp.ndarray = - nodes[i] / node_difference * polynomial_difference
                summand3: jnp.ndarray = jnp.roll(polynomial_difference / node_difference, 1, axis=0)

                return polynomials_inner.at[i].set(summand1 + summand2 + summand3)

            return jax.lax.fori_loop(k, n, inner_loop, polynomials_outer)

        return jax.lax.fori_loop(1, n, outer_loop, initial_polynomials)[n - 1]


    # def _internal_perform_action_(self) -> jnp.ndarray:
    #     data_type: DTypeLike = self._pipeline_data_[0].data_type
    #     nodes = self._pipeline_data_[0].interpolation_nodes.astype(data_type)
    #     values = self._pipeline_data_[0].interpolation_values.astype(data_type)
    #     n = nodes.size
    #
    #     initial_coefficients: jnp.ndarray = values.copy()[None, :]
    #
    #     def calc_node_differences(k: int) -> jnp.ndarray:
    #         end_nodes: jnp.ndarray = nodes[k:]
    #         start_nodes: jnp.ndarray = nodes[:-k]
    #         return end_nodes - start_nodes
    #
    #     def calc_polynomial_differences(polynomials: jnp.ndarray) -> jnp.ndarray:
    #         end_polynomials: jnp.ndarray = polynomials[:, 1:]
    #         start_polynomials: jnp.ndarray = polynomials[:, :-1]
    #         return end_polynomials - start_polynomials
    #
    #     def loop(k: int, old_polynomials: jnp.ndarray) -> jnp.ndarray:
    #         new_polynomials: jnp.ndarray = old_polynomials[:, 0:-1]
    #         node_differences: jnp.ndarray = calc_node_differences(k)
    #         polynomial_differences: jnp.ndarray = calc_polynomial_differences(old_polynomials)
    #         quotients: jnp.ndarray = polynomial_differences / node_differences
    #
    #         new_polynomials -= nodes[k:] * quotients
    #         new_polynomials += jnp.pad(quotients, ((1, 0), (0, 0)))  # Multiply polynomials with 'x'
    #         return new_polynomials
    #
    #     return jax.lax.fori_loop(1, n, loop, initial_coefficients)



    # @staticmethod
    # def _create_compiled_callable_(nodes: jnp.ndarray, interpolation_values: jnp.ndarray, data_type: DTypeLike) -> callable:
    #     def _internal_perform_action_() -> jnp.ndarray:
    #         # Number of interpolation points
    #         n = nodes.size - 1
    #
    #         initial_coefficients: jnp.ndarray = interpolation_values.copy()#((n, n), dtype=data_type)
    #
    #         def outer_loop(k: int, polynomials: jnp.ndarray) -> jnp.ndarray:
    #             new_polynomials: jnp.ndarray = jnp.empty((n + 1 - k, k + 1), dtype=data_type)
    #
    #
    #
    #             def inner_loop(i: int, polynomials) -> jnp.ndarray:
    #                 new_polynomials[i] = polynomials[i - k]
    #                 difference = polynomials[i] - polynomials[i - 1]
    #                 new_polynomials[i] -= difference * nodes[i] / (nodes[i] - nodes[i - k])
    #                 new_polynomials[i] += jnp.pad((difference / (nodes[i] - nodes[i - k])), (1, 0))
    #
    #         jax.lax.fori_loop(1, n + 1, outer_loop, initial_coefficients)
    #
    #
    #         return coefficients
    #
    #     return (
    #         jax.jit(_internal_perform_action_)  # → XLA-compatible HLO
    #         .lower()  # → Low-Level-IR
    #         .compile()  # → executable Binary
    #     )

