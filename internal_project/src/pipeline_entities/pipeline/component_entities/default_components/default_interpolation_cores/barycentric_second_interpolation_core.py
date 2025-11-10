import jax
import jax.numpy as jnp
from jax import block_until_ready

from jax.typing import DTypeLike

from functions.defaults.default_interpolants.barycentric_second_interpolant import BarycentricSecondInterpolant
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.interpolation_cores.barycentric_second_interpolation_core_meta_info import barycentric_second_interpolation_core_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from data_classes.pipeline_data.pipeline_data import PipelineData


@pipeline_component(id="barycentric2 interpolation", type=InterpolationCore, meta_info=barycentric_second_interpolation_core_meta_info)
class BarycentricSecondInterpolationCore(InterpolationCore):
    """
    Pipeline component that constructs the second form of the barycentric interpolation.
    It computes the barycentric weights and attaches the resulting interpolant to the pipeline data.
    """


    ###############################
    ### Attributes of instances ###
    ###############################
    _compiled_jax_callable_: callable


    ###################
    ### Constructor ###
    ###################
    def __init__(self, pipeline_data: list[PipelineData], additional_execution_data: AdditionalComponentExecutionData) -> None:
        """
        Initialize the barycentric second interpolation core.

        Args:
            pipeline_data (list[PipelineData]): Input pipeline data.
            additional_execution_data (AdditionalComponentExecutionData): Additional execution data.
        """

        super().__init__(pipeline_data, additional_execution_data)
        data_type: DTypeLike = pipeline_data[0].data_type
        nodes_dummy: jnp.ndarray = jnp.empty_like(pipeline_data[0].interpolation_nodes, dtype=data_type)
        self._compiled_jax_callable_ = (jax.jit(self._internal_perform_action_).lower(nodes_dummy).compile())


    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        """
        Compute barycentric weights, build the barycentric interpolant object, and attach it to the pipeline data.

        Returns:
            PipelineData: Updated pipeline data with the barycentric interpolant assigned.
        """

        pd: PipelineData = self._pipeline_data_[0]
        nodes: jnp.ndarray = pd.interpolation_nodes.astype(pd.data_type)
        weights: jnp.ndarray = self._compiled_jax_callable_(nodes)
        block_until_ready(weights)
        interpolant = BarycentricSecondInterpolant(name="Barycentric2", nodes=pd.interpolation_nodes, values=pd.interpolation_values, weights=weights)
        pd.interpolant = interpolant
        return pd


    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _internal_perform_action_(nodes: jnp.ndarray) -> jnp.ndarray:
        n: int = nodes.shape[0]
        initial_weights: jnp.ndarray = jnp.empty_like(nodes)

        def body(i: int, weights: jnp.ndarray) -> jnp.ndarray:
            diffs = nodes[i] - nodes
            diffs = diffs.at[i].set(1.0)
            wj = 1.0 / jnp.prod(diffs)
            return weights.at[i].set(wj)

        return jax.lax.fori_loop(0, n, body, initial_weights)
