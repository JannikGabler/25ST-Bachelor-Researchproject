import jax
import jax.numpy as jnp
from jax import block_until_ready
from jax.typing import DTypeLike

from functions.defaults.default_interpolants.aitken_neville_interpolant import AitkenNevilleInterpolant
from data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.interpolation_cores.aitken_neville_interpolation_core_meta_info import \
    aitken_neville_interpolation_core_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import \
    pipeline_component
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import \
    AdditionalComponentExecutionData


@pipeline_component(id="aitken neville interpolation", type=InterpolationCore, meta_info=aitken_neville_interpolation_core_meta_info)
class AitkenNevilleInterpolationCore(InterpolationCore):
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

        data_type: DTypeLike = pipeline_data[0].data_type

        nodes_dummy: jnp.ndarray = jnp.empty_like(pipeline_data[0].interpolation_nodes, dtype=data_type)
        values_dummy: jnp.ndarray = jnp.empty_like(pipeline_data[0].interpolation_values, dtype=data_type)

        self._compiled_jax_callable_ = jax.jit(self._internal_perform_action_).lower(nodes_dummy, values_dummy).compile()
        # print(f"Jaxpr {len(nodes_dummy)}:\n", jax.make_jaxpr(self._internal_perform_action_)(nodes_dummy, values_dummy))

    # def __init__(self, pipeline_data: list[PipelineData], additional_execution_data: AdditionalComponentExecutionData) -> None:
    #     super().__init__(pipeline_data, additional_execution_data)



    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        pipeline_data: PipelineData = self._pipeline_data_[0]

        data_type: DTypeLike = pipeline_data.data_type
        nodes = pipeline_data.interpolation_nodes.astype(data_type)
        values = pipeline_data.interpolation_values.astype(data_type)

        coefficients: jnp.ndarray = self._compiled_jax_callable_(nodes, values)
        block_until_ready(coefficients)

        interpolant = AitkenNevilleInterpolant("Aitken-Neville", coefficients)

        pipeline_data.interpolant = interpolant
        return pipeline_data

    # def perform_action(self) -> PipelineData:
    #     pipeline_data: PipelineData = self._pipeline_data_[0]
    #
    #     coefficients: jnp.ndarray = self._compiled_jax_callable_()
    #     interpolant = AitkenNevilleInterpolant("Aitken-Neville", coefficients)
    #
    #     pipeline_data.interpolant = interpolant
    #     return pipeline_data



    #######################
    ### Private methods ###
    #######################
    # @staticmethod
    # def _internal_perform_action_(nodes, values) -> jnp.ndarray:
    #     n: int = nodes.shape[0]
    #     dtype = values.dtype
    #     eps = jnp.finfo(dtype).eps
    #
    #     # 1) Anfangsmatrix (n x n): Spalte 0 = Werte
    #     polynomials = jnp.zeros((n, n), dtype=dtype).at[:, 0].set(values)
    #
    #     # 2) Präberechnungen mit statischen Shapes:
    #     # nodes_rolled_matrix[:, k] = nodes rolled by k  -> shape (n, n)
    #     nodes_rolled = jnp.stack([jnp.roll(nodes, k) for k in range(n)], axis=1)  # (n, n)
    #
    #     # denom_matrix[i, k] = nodes[i] - nodes[i-k]  (für k=0 ergibt 0, wir nutzen k>=1 später)
    #     denom_matrix = nodes[:, None] - nodes_rolled  # (n, n)
    #
    #     # mask_valid[i, k] = (i >= k)
    #     idx = jnp.arange(n)
    #     mask_valid = (idx[:, None] >= jnp.arange(n)[None, :])  # (n, n) boolean
    #
    #     # Ersetze in denom_matrix nahe 0 durch eps (sicherer Divisor)
    #     denom_safe_matrix = jnp.where(jnp.abs(denom_matrix) > eps, denom_matrix,
    #                                   jnp.sign(denom_matrix) * eps + eps)
    #
    #     # 3) Äußere Schleife über k: pro k benutzen wir die bereits berechneten Spalten k
    #     def outer_loop(k, polynomials_outer):
    #         # compute poly_diff = rows - rows_shifted (whole matrix)
    #         rows_shifted = jnp.roll(polynomials_outer, 1, axis=0)  # (n,n)
    #         poly_diff = polynomials_outer - rows_shifted  # (n,n)
    #
    #         # Spalte k des denom- und mask-Matrix auswählen (jeweils shape (n,))
    #         denom_k = denom_safe_matrix[:, k]  # (n,)
    #         mask_k = mask_valid[:, k]  # (n,)
    #
    #         denom_k_2d = denom_k[:, None]  # (n,1)
    #         mask_k_2d = mask_k[:, None]  # (n,1)
    #
    #         # Nur für i >= k wirken die Differenzen; maskiert für i<k
    #         poly_diff_masked = poly_diff * mask_k_2d
    #
    #         summand1 = polynomials_outer
    #         summand2 = - (nodes[:, None] / denom_k_2d) * poly_diff_masked
    #
    #         div = poly_diff_masked / denom_k_2d  # (n,n)
    #         # shifted: insert zero column at left, take div[:, :-1] to align degrees
    #         if n > 1:
    #             shifted = jnp.pad(div[:, :-1], ((0, 0), (1, 0)))
    #         else:
    #             shifted = jnp.zeros((n, 1), dtype=dtype)
    #
    #         new_rows_full = summand1 + summand2 + shifted  # (n,n)
    #         # Für i < k ist poly_diff_masked = 0 => new_rows_full[i] == summand1[i] (also unverändert)
    #
    #         return new_rows_full
    #
    #     polynomials_final = jax.lax.fori_loop(1, n, outer_loop, polynomials)
    #
    #     # letzte Zeile = Monom-Koeffizienten
    #     return polynomials_final[n - 1]

    @staticmethod
    def _internal_perform_action_(nodes: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
        n: int = len(nodes)

        # Initialize coefficients array with function values
        initial_polynomials = jnp.zeros((n, n)).at[:, 0].set(values)

        # Compute divided differences of increasing order
        def outer_loop(k, polynomials_outer):
            # Update entries for current order of divided differences
            def inner_loop(i, polynomials_inner):
                node_difference: jnp.ndarray = nodes[i] - nodes[i - k]
                polynomial_difference: jnp.ndarray = polynomials_outer[i] - polynomials_outer[i - 1]

                summand1: jnp.ndarray = polynomials_outer[i]
                summand2: jnp.ndarray = - nodes[i] / node_difference * polynomial_difference
                summand3: jnp.ndarray = jnp.roll(polynomial_difference / node_difference, 1, axis=0)

                return polynomials_inner.at[i].set(summand1 + summand2 + summand3)

            return jax.lax.fori_loop(k, n, inner_loop, polynomials_outer)

        return jax.lax.fori_loop(1, n, outer_loop, initial_polynomials)[n - 1]

    # def _internal_perform_action_(nodes, values) -> jnp.ndarray:
    #     data_type: DTypeLike = self._pipeline_data_[0].data_type
    #     nodes = self._pipeline_data_[0].interpolation_nodes.astype(data_type)
    #     values = self._pipeline_data_[0].interpolation_values.astype(data_type)
    #     n: int = self._pipeline_data_[0].node_count
    #
    #     # Initialize coefficients array with function values
    #     initial_polynomials = jnp.zeros((n, n)).at[:, 0].set(values)
    #
    #     # Compute divided differences of increasing order
    #     def outer_loop(k, polynomials_outer):
    #         # Update entries for current order of divided differences
    #         def inner_loop(i, polynomials_inner):
    #             node_difference: jnp.ndarray = nodes[i] - nodes[i - k]
    #             polynomial_difference: jnp.ndarray = polynomials_outer[i] - polynomials_outer[i - 1]
    #
    #             summand1: jnp.ndarray = polynomials_outer[i]
    #             summand2: jnp.ndarray = - nodes[i] / node_difference * polynomial_difference
    #             summand3: jnp.ndarray = jnp.roll(polynomial_difference / node_difference, 1, axis=0)
    #
    #             return polynomials_inner.at[i].set(summand1 + summand2 + summand3)
    #
    #         return jax.lax.fori_loop(k, n, inner_loop, polynomials_outer)
    #
    #     return jax.lax.fori_loop(1, n, outer_loop, initial_polynomials)[n - 1]



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

