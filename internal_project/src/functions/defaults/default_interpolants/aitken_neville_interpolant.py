import jax
import jax.numpy as jnp

from functions.abstracts.compilable_function import CompilableFunction


class AitkenNevilleInterpolant(CompilableFunction):
    """
    TODO
    """

    ###############################
    ### Attributes of instances ###
    ###############################
    _coefficients_: jnp.ndarray

    ###################
    ### Constructor ###
    ###################
    def __init__(self, name: str, coefficients: jnp.ndarray):
        super().__init__(name)
        self._coefficients_ = coefficients

    ##########################
    ### Overridden methods ###
    ##########################
    def _get_internal_evaluate_function_(self, **kwargs) -> callable:
        if "use_compensation" in kwargs and kwargs["use_compensation"] is True:
            return self._internal_evaluate_with_compensation_
        else:
            return self._internal_evaluate_without_compensation_

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(coefficients={repr(self._coefficients_)})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(coefficients={str(self._coefficients_)})"

    def __hash__(self):
        return hash(self._coefficients_)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        else:
            return jnp.array_equal(
                self._coefficients_, other._coefficients_, equal_nan=True
            ).item()

    #######################
    ### Private methods ###
    #######################
    def _internal_evaluate_without_compensation_(
        self, evaluation_points: jnp.ndarray
    ) -> jnp.ndarray:
        n: int = self._coefficients_.size
        coefficients: jnp.ndarray = self._coefficients_.astype(self._data_type_)
        initial_accumulator: jnp.ndarray = jnp.full_like(
            evaluation_points, coefficients[n - 1]
        )  # Already has the right dtype

        def horner_step(i: int, accumulator: jnp.ndarray) -> jnp.ndarray:
            index: int = n - i - 1
            return accumulator * evaluation_points + coefficients[index]

        return jax.lax.fori_loop(1, n, horner_step, initial_accumulator)

    def _internal_evaluate_with_compensation_(
        self, evaluation_points: jnp.ndarray
    ) -> jnp.ndarray:
        n: int = self._coefficients_.size
        coefficients: jnp.ndarray = self._coefficients_.astype(self._data_type_)
        initial_accumulator: jnp.ndarray = jnp.full_like(
            evaluation_points, coefficients[n - 1]
        )  # Already has the right dtype
        initial_error_accumulator: jnp.ndarray = jnp.zeros_like(evaluation_points)

        def mul(accumulator: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            p: jnp.ndarray = accumulator * evaluation_points
            p_error: jnp.ndarray = jax.lax.fma(accumulator, evaluation_points, -p)
            return p, p_error

        def addition(p: jnp.ndarray, index: int) -> tuple[jnp.ndarray, jnp.ndarray]:
            s: jnp.ndarray = p + coefficients[index]
            difference: jnp.ndarray = s - p
            s_error: jnp.ndarray = (p - (sum - difference)) + (
                coefficients[index] - difference
            )
            return s, s_error

        def horner_step(
            i: int, accumulators: tuple[jnp.ndarray, jnp.ndarray]
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            index: int = n - i - 1
            p, p_error = mul(accumulators[0])
            s, s_error = addition(p, index)

            new_error_accumulator: jnp.ndarray = accumulators[1] * evaluation_points + (
                p_error + s_error
            )
            return s, new_error_accumulator

        result: tuple[jnp.ndarray, jnp.ndarray] = jax.lax.fori_loop(
            1, n, horner_step, (initial_accumulator, initial_error_accumulator)
        )
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
