import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.interpolation_cores.chebyshev_interpolation_matrix_core_meta_info import (
    chebyshev_interpolation_matrix_core_meta_info,
)
from pipeline_entities.pipeline.component_entities.default_component_types.aotc_interpolation_core import (
    AOTCInterpolationCore,
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


@pipeline_component(
    id="chebyshev interpolation matrix",
    type=InterpolationCore,
    meta_info=chebyshev_interpolation_matrix_core_meta_info,
)
class ChebyshevInterpolationMatrixCore(AOTCInterpolationCore):
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

    ###################
    ### Constructor ###
    ###################
    def __init__(
        self,
        pipeline_data: list[PipelineData],
        additional_execution_data: AdditionalComponentExecutionData,
    ) -> None:
        super().__init__(pipeline_data, additional_execution_data)

    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        pipeline_data: PipelineData = self._pipeline_data_[0]

        interpolation_matrix: jnp.ndarray = self._compiled_jax_callable_()
        # Expose the matrix on the pipeline data object
        pipeline_data.interpolation_matrix = interpolation_matrix

        # (Optional) If your pipeline expects an "interpolant" object,
        # you could wrap this matrix elsewhere to solve M c = y.
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
        """
        Construct M with entries M[i, j] = T_j(x_i), j = 0..n-1.

        Implementation details:
        - Uses θ_i = arccos(x_i) and T_j(x_i) = cos(j * θ_i).
        - Fully JAX-compatible (jit/compile-ready).
        """
        data_type: DTypeLike = self._pipeline_data_[0].data_type
        x: jnp.ndarray = self._pipeline_data_[0].interpolation_nodes.astype(data_type)
        n: int = self._pipeline_data_[0].node_count  # number of nodes

        # Clip to [-1, 1] for numerical safety before arccos
        x_clipped = jnp.clip(x, -1.0, 1.0)
        theta = jnp.arccos(x_clipped)

        # Allocate matrix M (n x n): columns are T_j, j=0..n-1
        M0: jnp.ndarray = jnp.zeros((n, n), dtype=data_type)

        def fill_col(j: int, M: jnp.ndarray) -> jnp.ndarray:
            col = jnp.cos(j * theta)  # T_j(x) = cos(j arccos x)
            return M.at[:, j].set(col)

        M = jax.lax.fori_loop(0, n, fill_col, M0)
        return M
