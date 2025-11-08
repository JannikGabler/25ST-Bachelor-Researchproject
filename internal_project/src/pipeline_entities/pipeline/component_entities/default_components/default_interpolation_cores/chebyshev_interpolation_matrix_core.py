import jax
import jax.numpy as jnp
from jax import block_until_ready
from jax.typing import DTypeLike

from data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.interpolation_cores.chebyshev_interpolation_matrix_core_meta_info import chebyshev_interpolation_matrix_core_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData


@pipeline_component(id="chebyshev interpolation matrix", type=InterpolationCore, meta_info=chebyshev_interpolation_matrix_core_meta_info)
class ChebyshevInterpolationMatrixCore(InterpolationCore):
    """
    Pipeline component that constructs the Chebyshev (first-kind) interpolation matrix M for the interpolation nodes provided in the pipeline data.
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
        Initialize the Chebyshev interpolation matrix core.

        Args:
            pipeline_data (list[PipelineData]): Input pipeline data containing interpolation nodes.
            additional_execution_data (AdditionalComponentExecutionData): Additional execution data.

        Returns:
            None
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
        Compute the Chebyshev interpolation matrix for the current pipeline data
        and attach it to the PipelineData instance.

        Returns:
            PipelineData: Updated pipeline data with the computed interpolation matrix assigned.
        """

        pd: PipelineData = self._pipeline_data_[0]
        nodes: jnp.ndarray = pd.interpolation_nodes.astype(pd.data_type)
        interpolation_matrix: jnp.ndarray = self._compiled_jax_callable_(nodes)
        block_until_ready(interpolation_matrix)
        pd.interpolation_matrix = interpolation_matrix
        return pd


    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _internal_perform_action_(nodes: jnp.ndarray) -> jnp.ndarray:
        n: int = nodes.shape[0]  # number of nodes
        nodes_clipped = jnp.clip(nodes, -1.0, 1.0)
        theta = jnp.arccos(nodes_clipped)
        M0: jnp.ndarray = jnp.zeros((n, n), dtype=nodes.dtype)

        def fill_col(j: int, M: jnp.ndarray) -> jnp.ndarray:
            col = jnp.cos(j * theta)
            return M.at[:, j].set(col)

        M = jax.lax.fori_loop(0, n, fill_col, M0)
        return M
