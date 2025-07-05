import jax

from data_structures.interpolants.default_interpolants.chebyshev_interpolant import ChebyshevInterpolant
from pipeline_entities.component_meta_info.default_component_meta_infos.interpolation_cores.barycentric_first_interpolation_core_meta_info import \
    barycentric_first_interpolation_core_meta_info
from pipeline_entities.components.abstracts.interpolation_core import InterpolationCore
import jax.numpy as jnp

from pipeline_entities.components.decorators.pipeline_component import pipeline_component
from pipeline_entities.data_transfer.pipeline_data import PipelineData


@pipeline_component(id="Barycentric First Form Interpolation", type=InterpolationCore, meta_info=barycentric_first_interpolation_core_meta_info)
class EquidistantNodeGenerator(InterpolationCore):
    """
    Computes the barycentric weights for the first form of the barycentric interpolation formula.

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
    def __init__(self, pipeline_data: PipelineData) -> None:
        super().__init__(pipeline_data)

        nodes = pipeline_data.nodes
        self.function_values = pipeline_data.function_values

        self._compiled_jax_callable_ = self._create_compiled_callable_(nodes)




    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> None:
        weights = self._compiled_jax_callable_()
        values = self._pipeline_data_.function_values

        interpolant = ChebyshevInterpolant(
            nodes=self._pipeline_data_.nodes,
            values=values,
            weights=weights
        )

        self._pipeline_data_.interpolant = interpolant



    #######################
    ### Private methods ###
    #######################
    def _create_compiled_callable_(self, nodes: jnp.ndarray):

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