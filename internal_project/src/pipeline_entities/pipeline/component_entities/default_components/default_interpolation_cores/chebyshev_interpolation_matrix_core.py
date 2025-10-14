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
    Builds the Chebyshev (first-kind) interpolation/Vandermonde matrix M for the
    given interpolation nodes in the current PipelineData.

    Definition:
        M[i, j] = T_j(x_i),   j = 0..n-1, i = 0..n-1,
    where T_j is the Chebyshev polynomial of the first kind and x_i are the support nodes.

    Notes:
    - We assume the nodes are given on [-1, 1]. If your pipeline provides nodes on [a, b],
      map them beforehand or add a mapping step here.
    - The result is stored on the PipelineData as `interpolation_matrix`.
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

        self._compiled_jax_callable_ = (jax.jit(self._internal_perform_action_).lower(nodes_dummy).compile())


    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
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
        """
        Construct M with entries M[i, j] = T_j(x_i), j = 0..n-1.

        Implementation details:
        - Uses θ_i = arccos(x_i) and T_j(x_i) = cos(j * θ_i).
        - Fully JAX-compatible (jit/compile-ready).
        """

        n: int = nodes.shape[0]  # number of nodes

        nodes_clipped = jnp.clip(nodes, -1.0, 1.0)
        theta = jnp.arccos(nodes_clipped)

        M0: jnp.ndarray = jnp.zeros((n, n), dtype=nodes.dtype)

        def fill_col(j: int, M: jnp.ndarray) -> jnp.ndarray:
            col = jnp.cos(j * theta)
            return M.at[:, j].set(col)

        M = jax.lax.fori_loop(0, n, fill_col, M0)
        return M
