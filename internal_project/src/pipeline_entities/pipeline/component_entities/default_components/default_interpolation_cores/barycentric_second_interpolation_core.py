import jax
import jax.numpy as jnp

from jax.typing import DTypeLike

from interpolants.default_interpolants.barycentric_second_interpolant import BarycentricSecondInterpolant
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.interpolation_cores.barycentric_second_interpolation_core_meta_info import \
    barycentric_second_interpolation_core_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.aotc_interpolation_core import \
    AOTCInterpolationCore
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore

from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData


@pipeline_component(id="barycentric2 interpolation", type=InterpolationCore, meta_info=barycentric_second_interpolation_core_meta_info)
class BarycentricSecondInterpolationCore(AOTCInterpolationCore):
    """
    Computes the barycentric weights for the second form of the barycentric interpolation formula.

    Returns:
        1D array containing the barycentric weights.
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

        weights: jnp.ndarray = self._compiled_jax_callable_()

        interpolant = BarycentricSecondInterpolant(
            name="Barycentric2",
            nodes=pipeline_data.interpolation_nodes,
            values=pipeline_data.interpolation_values,
            weights=weights
        )

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
        nodes: jnp.ndarray = self._pipeline_data_[0].interpolation_nodes.astype(data_type)
        n: int = self._pipeline_data_[0].node_count

        initial_weights: jnp.ndarray = jnp.empty_like(nodes)

        def body(i: int, weights: jnp.ndarray) -> jnp.ndarray:
            # 1D-Array aller Differenzen x_j - x_k
            diffs = nodes[i] - nodes
            # Setze die eigene Differenz auf 1, damit sie das Produkt nicht beeinflusst
            diffs = diffs.at[i].set(1.0)
            # Produkt bilden und invertieren
            wj = 1.0 / jnp.prod(diffs)
            return weights.at[i].set(wj)

        return jax.lax.fori_loop(0, n, body, initial_weights)



    # def _internal_perform_action_(self) -> jnp.ndarray:
    #     data_type: DTypeLike = self._pipeline_data_[0].data_type
    #     nodes: jnp.ndarray = self._pipeline_data_[0].interpolation_nodes.astype(data_type)
    #
    #     # Create a square matrix where each entry [j, k] is the difference between node j and node k
    #     # Note that the diagonal entries are all zero, since each node is subtracted from itself
    #     pairwise_diff: jnp.ndarray = nodes[:, None] - nodes[None, :]
    #
    #     # Create a boolean matrix with False on the diagonal and True elsewhere
    #     # This is used to exclude self-differences (which are zero) from the product
    #     bool_diff: jnp.ndarray = ~jnp.eye(len(nodes), dtype=bool)
    #
    #     # Replace diagonal entries (which are zero) with 1.0 to avoid affecting the product
    #     # Then compute the product across each row (over all k ≠ j)
    #     product: jnp.ndarray = jnp.prod(jnp.where(bool_diff, pairwise_diff, 1.0), axis=1)
    #
    #     # Divide 1.0 by the product to get the barycentric weights (Equation (5.6))
    #     return 1.0 / product