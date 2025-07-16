import jax

from interpolants.default_interpolants.barycentric_second_interpolant import BarycentricSecondInterpolant
from pipeline_entities.pipeline.component_entities.component_meta_info.default_component_meta_infos.interpolation_cores.barycentric_second_interpolation_core_meta_info import \
    barycentric_second_interpolation_core_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore
import jax.numpy as jnp

from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData


@pipeline_component(id="barycentric2 interpolation", type=InterpolationCore, meta_info=barycentric_second_interpolation_core_meta_info)
class BarycentricSecondInterpolationCore(InterpolationCore):
    """
    Computes the barycentric weights for the second form of the barycentric interpolation formula.

    Returns:
        1D array containing the barycentric weights.
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

        if data.data_type is not None:
            data.interpolation_nodes = data.interpolation_nodes.astype(data.data_type)
            data.interpolation_values = data.interpolation_values.astype(data.data_type)

        self._compiled_jax_callable_ = self._create_compiled_callable_(data.interpolation_nodes)



    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        pipeline_data: PipelineData = self._pipeline_data_[0]

        weights: jnp.ndarray = self._compiled_jax_callable_()

        interpolant = BarycentricSecondInterpolant(
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
    def _create_compiled_callable_(nodes: jnp.ndarray):

        def _internal_perform_action_() -> jnp.ndarray:
            # Create a square matrix where each entry [j, k] is the difference between node j and node k
            # Note that the diagonal entries are all zero, since each node is subtracted from itself
            pairwise_diff = nodes[:, None] - nodes[None, :]

            # Create a boolean matrix with False on the diagonal and True elsewhere
            # This is used to exclude self-differences (which are zero) from the product
            bool_diff = ~jnp.eye(len(nodes), dtype=bool)

            # Replace diagonal entries (which are zero) with 1.0 to avoid affecting the product
            # Then compute the product across each row (over all k ≠ j)
            product = jnp.prod(jnp.where(bool_diff, pairwise_diff, 1.0), axis=1)

            # Divide 1.0 by the product to get the barycentric weights (Equation (5.6))
            return 1.0 / product

        return (
            jax.jit(_internal_perform_action_)       # → XLA-compatible HLO
                .lower()    # → Low-Level-IR
                .compile()  # → executable Binary
        )