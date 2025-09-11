import jax
import jax.numpy as jnp
from jax import block_until_ready

from data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.evaluation_components.aitken_neville_evaluator_meta_info import \
    aitken_neville_evaluator_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.evaluator_component import EvaluatorComponent
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import \
    pipeline_component
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import \
    AdditionalComponentExecutionData


@pipeline_component(id="aitken neville evaluator", type=EvaluatorComponent, meta_info=aitken_neville_evaluator_meta_info)
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
    def __init__(self, pipeline_data: list[PipelineData], additional_execution_data: AdditionalComponentExecutionData) -> None:
        super().__init__(pipeline_data, additional_execution_data)

        data_type = self._pipeline_data_[0].data_type

        nodes_dummy = jnp.empty_like(pipeline_data[0].interpolation_nodes, dtype=data_type)
        values_dummy = jnp.empty_like(pipeline_data[0].interpolation_values, dtype=data_type)
        evaluation_points_dummy = jnp.empty_like(pipeline_data[0].interpolant_evaluation_points, dtype=data_type)

        self._compiled_jax_callable_ = jax.jit(self._internal_perform_action_).lower(nodes_dummy, values_dummy, evaluation_points_dummy).compile()

    # def __init__(self, pipeline_data: list[PipelineData], additional_execution_data: AdditionalComponentExecutionData) -> None:
    #     super().__init__(pipeline_data, additional_execution_data)
    #
    #     self._compiled_jax_callable_ = jax.jit(self._internal_perform_action_).lower().compile()



    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        pipeline_data: PipelineData = self._pipeline_data_[0]

        data_type = pipeline_data.data_type
        nodes = pipeline_data.interpolation_nodes.astype(data_type)
        values = pipeline_data.interpolation_values.astype(data_type)
        evaluation_points = pipeline_data.interpolant_evaluation_points.astype(data_type)

        interpolant_values = self._compiled_jax_callable_(nodes, values, evaluation_points)
        block_until_ready(interpolant_values)

        pipeline_data.interpolant_values = interpolant_values
        return pipeline_data

    # def perform_action(self) -> PipelineData:
    #     pipeline_data: PipelineData = self._pipeline_data_[0]
    #
    #     pipeline_data.interpolant_values = self._compiled_jax_callable_()
    #
    #     return pipeline_data



    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _internal_perform_action_(nodes, values, evaluation_points):
        # data_type = self._pipeline_data_[0].data_type
        n = len(nodes)

        # evaluator für genau einen auswertungspunkt x (arbeitet nur auf 1D arrays der länge n)
        def eval_one(x):
            # initial: P_i^0 = f(x_i)  (1D array shape (n,))
            pol = values.copy()

            def outer_loop(k, pol_array):
                # update pol_array[i] for i=k..n-1 in-place-logischerweise (funktional via .at)
                def inner(i, p):
                    diff = p[i] - p[i - 1]  # shape: scalar
                    q = (x - nodes[i]) / (nodes[i] - nodes[i - k])  # scalar
                    return p.at[i].set(p[i] + q * diff)

                # fori_loop von i=k bis n-1
                return jax.lax.fori_loop(k, n, inner, pol_array)

            # fori_loop über k=1..n-1
            final_pol = jax.lax.fori_loop(1, n, outer_loop, pol)
            # der Wert des Interpolanten ist final_pol[n-1]
            return final_pol[n - 1]

        # vectorize über alle evaluation_points: result shape (m,)
        vmapped = jax.vmap(eval_one)
        return vmapped(evaluation_points)


    # def _internal_perform_action_(self) -> jnp.ndarray:
    #     data_type: DTypeLike = self._pipeline_data_[0].data_type
    #     nodes = self._pipeline_data_[0].interpolation_nodes.astype(data_type)
    #     values = self._pipeline_data_[0].interpolation_values.astype(data_type)
    #     evaluation_points = self._pipeline_data_[0].interpolant_evaluation_points.astype(data_type)
    #     n: int = self._pipeline_data_[0].node_count
    #     m: int = len(evaluation_points)
    #
    #     # Initialize values array with function values
    #     initial_values = values[:, None] * jnp.ones((1, m))
    #
    #     def outer_loop(k, polynomials_outer):
    #
    #         def inner_loop(i, polynomials_inner):
    #             value_differences = polynomials_outer[i] - polynomials_outer[i - 1]
    #             quotients = (evaluation_points - nodes[i]) / (nodes[i] - nodes[i - k])
    #             return polynomials_inner.at[i].set(polynomials_outer[i] + quotients * value_differences)
    #
    #         return jax.lax.fori_loop(k, n, inner_loop, polynomials_outer)
    #
    #     return jax.lax.fori_loop(1, n, outer_loop, initial_values)[n - 1]


        # # Compute divided differences of increasing order
        # def outer_loop(k, polynomials_outer):
        #     # Update entries for current order of divided differences
        #     def inner_loop(i, polynomials_inner):
        #         node_difference: jnp.ndarray = nodes[i] - nodes[i - k]
        #         polynomial_difference: jnp.ndarray = polynomials_outer[i] - polynomials_outer[i - 1]
        #
        #         summand1: jnp.ndarray = polynomials_outer[i]
        #         summand2: jnp.ndarray = - nodes[i] / node_difference * polynomial_difference
        #         summand3: jnp.ndarray = jnp.roll(polynomial_difference / node_difference, 1, axis=0)
        #
        #         return polynomials_inner.at[i].set(summand1 + summand2 + summand3)
        #
        #     return jax.lax.fori_loop(k, n, inner_loop, polynomials_outer)
        #
        # return jax.lax.fori_loop(1, n, outer_loop, initial_polynomials)[n - 1]


    # def _internal_perform_action_(self) -> jnp.ndarray:
    #     data_type: DTypeLike = self._pipeline_data_[0].data_type
    #     nodes = self._pipeline_data_[0].interpolation_nodes.astype(data_type)
    #     values = self._pipeline_data_[0].interpolation_values.astype(data_type)
    #     n = nodes.size
    #
    #     initial_coefficients: jnp.ndarray = values.copy()[None, :]
    #
    #     def calc_node_differences(k: int) -> jnp.ndarray:
    #         end_nodes: jnp.ndarray = nodes[k:]
    #         start_nodes: jnp.ndarray = nodes[:-k]
    #         return end_nodes - start_nodes
    #
    #     def calc_polynomial_differences(polynomials: jnp.ndarray) -> jnp.ndarray:
    #         end_polynomials: jnp.ndarray = polynomials[:, 1:]
    #         start_polynomials: jnp.ndarray = polynomials[:, :-1]
    #         return end_polynomials - start_polynomials
    #
    #     def loop(k: int, old_polynomials: jnp.ndarray) -> jnp.ndarray:
    #         new_polynomials: jnp.ndarray = old_polynomials[:, 0:-1]
    #         node_differences: jnp.ndarray = calc_node_differences(k)
    #         polynomial_differences: jnp.ndarray = calc_polynomial_differences(old_polynomials)
    #         quotients: jnp.ndarray = polynomial_differences / node_differences
    #
    #         new_polynomials -= nodes[k:] * quotients
    #         new_polynomials += jnp.pad(quotients, ((1, 0), (0, 0)))  # Multiply polynomials with 'x'
    #         return new_polynomials
    #
    #     return jax.lax.fori_loop(1, n, loop, initial_coefficients)



    # @staticmethod
    # def _create_compiled_callable_(nodes: jnp.ndarray, interpolation_values: jnp.ndarray, data_type: DTypeLike) -> callable:
    #     def _internal_perform_action_() -> jnp.ndarray:
    #         # Number of interpolation points
    #         n = nodes.size - 1
    #
    #         initial_coefficients: jnp.ndarray = interpolation_values.copy()#((n, n), dtype=data_type)
    #
    #         def outer_loop(k: int, polynomials: jnp.ndarray) -> jnp.ndarray:
    #             new_polynomials: jnp.ndarray = jnp.empty((n + 1 - k, k + 1), dtype=data_type)
    #
    #
    #
    #             def inner_loop(i: int, polynomials) -> jnp.ndarray:
    #                 new_polynomials[i] = polynomials[i - k]
    #                 difference = polynomials[i] - polynomials[i - 1]
    #                 new_polynomials[i] -= difference * nodes[i] / (nodes[i] - nodes[i - k])
    #                 new_polynomials[i] += jnp.pad((difference / (nodes[i] - nodes[i - k])), (1, 0))
    #
    #         jax.lax.fori_loop(1, n + 1, outer_loop, initial_coefficients)
    #
    #
    #         return coefficients
    #
    #     return (
    #         jax.jit(_internal_perform_action_)  # → XLA-compatible HLO
    #         .lower()  # → Low-Level-IR
    #         .compile()  # → executable Binary
    #     )

