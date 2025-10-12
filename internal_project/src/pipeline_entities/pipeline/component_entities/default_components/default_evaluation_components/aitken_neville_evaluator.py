import jax
import jax.numpy as jnp
from jax import block_until_ready

from data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.evaluation_components.aitken_neville_evaluator_meta_info import (
    aitken_neville_evaluator_meta_info,
)
from pipeline_entities.pipeline.component_entities.default_component_types.evaluator_component import (
    EvaluatorComponent,
)
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import (
    pipeline_component,
)
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import (
    AdditionalComponentExecutionData,
)


@pipeline_component(
    id="aitken neville evaluator",
    type=EvaluatorComponent,
    meta_info=aitken_neville_evaluator_meta_info,
)
class AitkenNevilleEvaluator(EvaluatorComponent):
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
    def __init__(
        self,
        pipeline_data: list[PipelineData],
        additional_execution_data: AdditionalComponentExecutionData,
    ) -> None:
        super().__init__(pipeline_data, additional_execution_data)

        data_type = self._pipeline_data_[0].data_type

        nodes_dummy = jnp.empty_like(
            pipeline_data[0].interpolation_nodes, dtype=data_type
        )
        values_dummy = jnp.empty_like(
            pipeline_data[0].interpolation_values, dtype=data_type
        )
        evaluation_points_dummy = jnp.empty_like(
            pipeline_data[0].interpolant_evaluation_points, dtype=data_type
        )

        self._compiled_jax_callable_ = (
            jax.jit(self._internal_perform_action_)
            .lower(nodes_dummy, values_dummy, evaluation_points_dummy)
            .compile()
        )

    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        pipeline_data: PipelineData = self._pipeline_data_[0]

        data_type = pipeline_data.data_type
        nodes = pipeline_data.interpolation_nodes.astype(data_type)
        values = pipeline_data.interpolation_values.astype(data_type)
        evaluation_points = pipeline_data.interpolant_evaluation_points.astype(
            data_type
        )

        interpolant_values = self._compiled_jax_callable_(
            nodes, values, evaluation_points
        )
        block_until_ready(interpolant_values)

        pipeline_data.interpolant_values = interpolant_values
        return pipeline_data

    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _internal_perform_action_(nodes, values, evaluation_points):
        # data_type = self._pipeline_data_[0].data_type
        n = len(nodes)

        # evaluator for exactly one evaluation point x (only for 1D arrays of length n)
        def eval_one(x):
            # initially: P_i^0 = f(x_i)  (1D array shape (n,))
            pol = values.copy()

            def outer_loop(k, pol_array):
                # update pol_array[i] for i=k..n-1 in-place (functional via .at)
                def inner(i, p):
                    diff = p[i] - p[i - 1]  # shape: scalar
                    q = (x - nodes[i]) / (nodes[i] - nodes[i - k])  # scalar
                    return p.at[i].set(p[i] + q * diff)

                # fori_loop i=k to n-1
                return jax.lax.fori_loop(k, n, inner, pol_array)

            # fori_loop k=1..n-1
            final_pol = jax.lax.fori_loop(1, n, outer_loop, pol)
            # value of the interpolant is final_pol[n-1]
            return final_pol[n - 1]

        # vectorize over all evaluation_points: result shape (m,)
        vmapped = jax.vmap(eval_one)
        return vmapped(evaluation_points)
