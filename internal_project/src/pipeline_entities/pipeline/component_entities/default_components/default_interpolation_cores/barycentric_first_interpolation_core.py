import jax
import jax.numpy as jnp
from jax import block_until_ready

from functions.defaults.default_interpolants.barycentric_first_interpolant import (
    BarycentricFirstInterpolant,
)
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.interpolation_cores.barycentric_first_interpolation_core_meta_info import (
    barycentric_first_interpolation_core_meta_info,
)
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import (
    InterpolationCore,
)


from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import (
    pipeline_component,
)
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import (
    AdditionalComponentExecutionData,
)
from data_classes.pipeline_data.pipeline_data import PipelineData


@pipeline_component(
    id="barycentric1 interpolation",
    type=InterpolationCore,
    meta_info=barycentric_first_interpolation_core_meta_info,
)
class BarycentricFirstInterpolationCore(InterpolationCore):
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
    def __init__(
        self,
        pipeline_data: list[PipelineData],
        additional_execution_data: AdditionalComponentExecutionData,
    ) -> None:
        super().__init__(pipeline_data, additional_execution_data)

        data_type = pipeline_data[0].data_type

        nodes_dummy = jnp.empty_like(
            pipeline_data[0].interpolation_nodes, dtype=data_type
        )

        self._compiled_jax_callable_ = (
            jax.jit(self._internal_perform_action_).lower(nodes_dummy).compile()
        )

    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        pd: PipelineData = self._pipeline_data_[0]

        nodes: jnp.ndarray = pd.interpolation_nodes.astype(pd.data_type)

        weights: jnp.ndarray = self._compiled_jax_callable_(nodes)
        block_until_ready(weights)

        interpolant = BarycentricFirstInterpolant(
            name="Barycentric1",
            nodes=pd.interpolation_nodes,
            values=pd.interpolation_values,
            weights=weights,
        )

        pd.interpolant = interpolant

        return pd

    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _internal_perform_action_(nodes) -> jnp.ndarray:
        n: int = nodes.shape[0]

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
