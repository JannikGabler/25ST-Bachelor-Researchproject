from typing import Any

import jax
import jax.numpy as jnp

from interpolants.abstracts.interpolant import Interpolant


class AitkenNevilleInterpolant(Interpolant):
    ###############################
    ### Attributes of instances ###
    ###############################
    _coefficients_: jnp.ndarray
    # _values_: jnp.ndarray
    # _nodes_: jnp.ndarray

    _use_compensated_horner_scheme_: bool



    ###################
    ### Constructor ###
    ###################
    def __init__(self, coefficients: jnp.ndarray, use_compensated_horner_scheme: bool = False):
        super().__init__()

        self._coefficients_ = coefficients
        self._use_compensated_horner_scheme_ = use_compensated_horner_scheme



    ##########################
    ### Overridden methods ###
    ##########################
    def _get_internal_evaluate_function_(self):
        if self._is_data_type_overridden_:
            if self._use_compensated_horner_scheme_:
                return self._internal_evaluate_with_data_type_overriding_with_compensation_
            else:
                return self._internal_evaluate_with_data_type_overriding_without_compensation_
        else:
            if self._use_compensated_horner_scheme_:
                return self._internal_evaluate_without_data_type_overriding_with_compensation_
            else:
                return self._internal_evaluate_without_data_type_overriding_without_compensation_



    def _is_data_type_overriding_required_(self) -> bool:
        return False

    def _get_data_type_for_no_overriding_(self) -> jnp.dtype:
        return self._coefficients_.dtype



    def __repr__(self) -> str:
        return f"AitkenNevilleInterpolant(coefficients={self._coefficients_})"

    def __str__(self) -> str:
        return self.__repr__()



    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, AitkenNevilleInterpolant):
            return False
        else:
            return jnp.array_equal(self._coefficients_, other._coefficients_).item()



    #######################
    ### Private methods ###
    #######################
    def _internal_evaluate_without_data_type_overriding_without_compensation_(self, evaluation_points: jnp.ndarray) -> jnp.ndarray:
        n: int = self._coefficients_.size
        initial_accumulator: jnp.ndarray = jnp.full_like(evaluation_points, self._coefficients_[n - 1]) # Already has the right dtype

        def horner_step(i: int, accumulator: jnp.ndarray) -> jnp.ndarray:
            index: int = n - i - 1
            return accumulator * evaluation_points + self._coefficients_[index]

        return jax.lax.fori_loop(1, n, horner_step, initial_accumulator)



    def _internal_evaluate_with_data_type_overriding_without_compensation_(self, evaluation_points: jnp.ndarray) -> jnp.ndarray:
        n: int = self._coefficients_.size
        coefficients: jnp.ndarray = self._coefficients_.astype(self._required_data_type_)
        initial_accumulator: jnp.ndarray = jnp.full_like(evaluation_points, coefficients[n - 1]) # Already has the right dtype

        def horner_step(i: int, accumulator: jnp.ndarray) -> jnp.ndarray:
            index: int = n - i - 1
            return accumulator * evaluation_points + coefficients[index]

        return jax.lax.fori_loop(1, n, horner_step, initial_accumulator)



    def _internal_evaluate_without_data_type_overriding_with_compensation_(self, evaluation_points: jnp.ndarray) -> jnp.ndarray:
        n: int = self._coefficients_.size
        initial_accumulator: jnp.ndarray = jnp.full_like(evaluation_points, self._coefficients_[n - 1]) # Already has the right dtype
        initial_error_accumulator: jnp.ndarray = jnp.zeros_like(evaluation_points)

        def mul(accumulator: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            p: jnp.ndarray = accumulator * evaluation_points
            p_error: jnp.ndarray = jax.lax.fma(accumulator, evaluation_points, -p)
            return p, p_error

        def addition(p: jnp.ndarray, index: int) -> tuple[jnp.ndarray, jnp.ndarray]:
            s: jnp.ndarray = p + self._coefficients_[index]
            difference: jnp.ndarray = s - p
            s_error: jnp.ndarray = (p - (sum - difference)) + (self._coefficients_[index] - difference)
            return s, s_error

        def horner_step(i: int, accumulators: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
            index: int = n - i - 1
            p, p_error = mul(accumulators[0])
            s, s_error = addition(p, index)

            new_error_accumulator: jnp.ndarray = accumulators[1] * evaluation_points + (p_error + s_error)
            return s, new_error_accumulator

        result: tuple[jnp.ndarray, jnp.ndarray] = jax.lax.fori_loop(1, n, horner_step, (initial_accumulator, initial_error_accumulator))
        return result[0] + result[1]



    def _internal_evaluate_with_data_type_overriding_with_compensation_(self, evaluation_points: jnp.ndarray) -> jnp.ndarray:
        n: int = self._coefficients_.size
        coefficients: jnp.ndarray = self._coefficients_.astype(self._required_data_type_)
        initial_accumulator: jnp.ndarray = jnp.full_like(evaluation_points, coefficients[n - 1]) # Already has the right dtype
        initial_error_accumulator: jnp.ndarray = jnp.zeros_like(evaluation_points)

        def mul(accumulator: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            p: jnp.ndarray = accumulator * evaluation_points
            p_error: jnp.ndarray = jax.lax.fma(accumulator, evaluation_points, -p)
            return p, p_error

        def addition(p: jnp.ndarray, index: int) -> tuple[jnp.ndarray, jnp.ndarray]:
            s: jnp.ndarray = p + coefficients[index]
            difference: jnp.ndarray = s - p
            s_error: jnp.ndarray = (p - (sum - difference)) + (coefficients[index] - difference)
            return s, s_error

        def horner_step(i: int, accumulators: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
            index: int = n - i - 1
            p, p_error = mul(accumulators[0])
            s, s_error = addition(p, index)

            new_error_accumulator: jnp.ndarray = accumulators[1] * evaluation_points + (p_error + s_error)
            return s, new_error_accumulator

        result: tuple[jnp.ndarray, jnp.ndarray] = jax.lax.fori_loop(1, n, horner_step, (initial_accumulator, initial_error_accumulator))
        return result[0] + result[1]


    # def aitken_neville_scalar(self, nodes: jnp.ndarray, values: jnp.ndarray, x: float) -> jnp.ndarray:
    #     """
    #     Computes interpolating polynomial at x
    #
    #     Args:
    #         nodes: x coordinates of the given points
    #         values: y coordinates of the given points
    #         x: the point at which to evaluate the polynomial
    #
    #     Returns:
    #         Interpolated value P(x)
    #     """
    #     m = nodes.shape[0]
    #     # (upper) triangular matrix
    #     P = jnp.zeros((m, m), dtype=values.dtype)
    #     # diagonal entries
    #     P = P.at[jnp.diag_indices(m)].set(values)
    #
    #     # avoid loops
    #     # outer loop over interval lengths k = 1, ..., m-1
    #     def body_k(k, P):
    #         # inner loop over starting indices i = 0, ..., m-k-1
    #         def body_i(i, P):
    #             j = i + k
    #             # mathematical formula for Aitken-Neville interpolation
    #             numerator = (x - nodes[i]) * P[i + 1, j] - (x - nodes[j]) * P[i, j - 1]
    #             denominator = nodes[j] - nodes[i]
    #             return P.at[i, j].set(numerator / denominator)
    #
    #         upper = jnp.int32(m - k)
    #         zero = jnp.int32(0)
    #         return jax.lax.fori_loop(zero, upper, body_i, P)
    #
    #     one = jnp.int32(1)
    #     P = jax.lax.fori_loop(one, m, body_k, P)
    #     # top right triangle is the result
    #     return P[0, m - 1]
    #
    #
    #
    # def evaluate(self, x:jnp.ndarray) -> jnp.ndarray:
    #     return jax.jit(jax.vmap(lambda x_i: self.aitken_neville_scalar(self._nodes_, self._values_, x_i)))(x)