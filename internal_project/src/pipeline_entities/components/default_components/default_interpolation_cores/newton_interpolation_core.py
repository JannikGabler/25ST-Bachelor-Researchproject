import jax

from data_structures.interpolants.default_interpolants.newton_interpolant import NewtonInterpolant
from pipeline_entities.component_meta_info.default_component_meta_infos.interpolation_cores.newton_interpolation_core_meta_info import \
    newton_interpolation_core_meta_info
from pipeline_entities.components.abstracts.interpolation_core import InterpolationCore
import jax.numpy as jnp

from pipeline_entities.components.decorators.pipeline_component import pipeline_component
from pipeline_entities.data_transfer.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.data_transfer.pipeline_data import PipelineData


@pipeline_component(id="newton interpolation", type=InterpolationCore, meta_info=newton_interpolation_core_meta_info)
class NewtonInterpolationCore(InterpolationCore):
    """
    Computes the coefficients of the Newton interpolation polynomial using divided differences.
    These coefficients can then be used to evaluate the interpolation polynomial efficiently in Newton form.

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
        data: PipelineData = pipeline_data[0]

        nodes: jnp.ndarray = data.interpolation_nodes
        interpolation_values: jnp.ndarray = data.interpolation_values

        self._compiled_jax_callable_ = self._create_compiled_callable_(nodes, interpolation_values)



    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        pipeline_data: PipelineData = self._pipeline_data_[0]

        weights = self._compiled_jax_callable_()   # TODO: add type

        interpolant = NewtonInterpolant(
            nodes=pipeline_data.interpolation_nodes,
            values=pipeline_data.interpolation_values,
            weights=weights
        )

        pipeline_data.interpolant = interpolant
        return pipeline_data



    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _create_compiled_callable_(nodes: jnp.ndarray, interpolation_values: jnp.ndarray) -> callable:

        def _internal_perform_action_() -> jnp.ndarray:
            # Number of interpolation points
            n = nodes.size

            # Initialize coefficients array with function values
            coefficients = interpolation_values.copy()

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

        return (
            jax.jit(_internal_perform_action_)       # → XLA-compatible HLO
                .lower()    # → Low-Level-IR
                .compile()  # → executable Binary
        )