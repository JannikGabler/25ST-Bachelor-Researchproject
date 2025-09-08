import jax
import jax.numpy as jnp
from jax import block_until_ready

from jax.typing import DTypeLike

from functions.defaults.default_interpolants.barycentric_second_interpolant import BarycentricSecondInterpolant
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.interpolation_cores.barycentric_second_interpolation_core_meta_info import \
    barycentric_second_interpolation_core_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore

from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData


@pipeline_component(id="barycentric2 chebyshev interpolation", type=InterpolationCore, meta_info=barycentric_second_interpolation_core_meta_info)
class BarycentricSecondChebyshevInterpolationCore(InterpolationCore):
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

        self._compiled_jax_callable_ = jax.jit(self._internal_perform_action_).lower().compile()



    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        pd: PipelineData = self._pipeline_data_[0]

        weights: jnp.ndarray = self._compiled_jax_callable_()
        block_until_ready(weights)

        interpolant = BarycentricSecondInterpolant(
            name="Barycentric2 Chebyshev",
            nodes=pd.interpolation_nodes,
            values=pd.interpolation_values,
            weights=weights
        )

        pd.interpolant = interpolant
        return pd



    #######################
    ### Private methods ###
    #######################
    def _internal_perform_action_(self) -> jnp.ndarray:
        data_type: DTypeLike = self._pipeline_data_[0].data_type
        node_count: int = self._pipeline_data_[0].node_count

        indices: jnp.ndarray = jnp.arange(node_count, dtype=jnp.int32)

        one = jnp.astype(1.0, data_type)
        minus_one = jnp.astype(-1.0, data_type)
        two = jnp.astype(2.0, data_type)

        weights: jnp.ndarray = jnp.where(indices % 2 == 0, one, minus_one)

        weights = weights.at[0].divide(two)
        weights = weights.at[-1].divide(two)

        return weights