from typing import Any

from exceptions.invalid_argument_exception import InvalidArgumentException
from interpolants.abstracts.interpolant import Interpolant
import jax
import jax.numpy as jnp


class BarycentricSecondInterpolant(Interpolant):
    ###############################
    ### Attributes of instances ###
    ###############################
    _nodes_: jnp.ndarray
    _values_: jnp.ndarray
    _weights_: jnp.ndarray



    ###################
    ### Constructor ###
    ###################
    def __init__(self, nodes: jnp.ndarray, values: jnp.ndarray, weights: jnp.ndarray):
        super().__init__()

        if len({nodes.shape, values.shape, weights.shape}) >= 2:
            raise InvalidArgumentException("The shapes of the given nodes, values and weight arrays differ, although"
               f"they're required to be equal (shape of nodes: {nodes.shape}, shape of values: {values.shape}, shape of"
               f"weights: {weights.shape}).")

        self._nodes_ = nodes
        self._values_ = values
        self._weights_ = weights



    ##########################
    ### Overridden methods ###
    ##########################
    def _get_internal_evaluate_function_(self) -> callable:
        if self._is_data_type_overridden_:
            return self._internal_evaluate_with_data_type_overriding_
        else:
            return self._internal_evaluate_without_data_type_overriding_



    def _is_data_type_overriding_required_(self) -> bool:
        return len({self._nodes_.dtype, self._values_.dtype, self._weights_.dtype}) >= 2

    def _get_data_type_for_no_overriding_(self) -> jnp.dtype:
        return self._nodes_.dtype



    def __repr__(self) -> str:
        return f"BarycentricSecondInterpolant(nodes={self._nodes_}, values={self._values_}, weights={self._weights_})"

    def __str__(self) -> str:
        return self.__repr__()



    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BarycentricSecondInterpolant):
            return False
        else:
            return (jnp.array_equal(self._nodes_, other._nodes_).item()
                    and jnp.array_equal(self._values_, other._values_).item()
                    and jnp.array_equal(self._weights_, other._weights_).item())



    #######################
    ### Private methods ###
    #######################
    def _internal_evaluate_without_data_type_overriding_(self, evaluation_points: jnp.ndarray) -> jnp.ndarray:
        def _evaluate_single_(point):
            differences: jnp.ndarray = point - self._nodes_

            exact_matches: jnp.ndarray = (differences == 0.0)
            exact_match_index: jnp.ndarray = jnp.argmax(exact_matches)
            exact_match_value: jnp.ndarray = self._values_[exact_match_index]

            return jnp.where(
                jnp.any(exact_matches),
                exact_match_value,
                jnp.sum((self._weights_ * self._values_) / differences) / jnp.sum(self._weights_ / differences)
            )

        return jax.vmap(_evaluate_single_)(evaluation_points)



    def _internal_evaluate_with_data_type_overriding_(self, evaluation_points: jnp.ndarray) -> jnp.ndarray:
        nodes: jnp.ndarray = self._nodes_.astype(self._required_data_type_)
        values: jnp.ndarray = self._values_.astype(self._required_data_type_)
        weights: jnp.ndarray = self._weights_.astype(self._required_data_type_)

        def _evaluate_single_(point):
            differences: jnp.ndarray = point - nodes

            exact_matches: jnp.ndarray = (differences == 0.0)
            exact_match_index: jnp.ndarray = jnp.argmax(exact_matches)
            exact_match_value: jnp.ndarray = values[exact_match_index]

            return jnp.where(
                jnp.any(exact_matches),
                exact_match_value,
                jnp.sum((weights * values) / differences) / jnp.sum(weights / differences)
            )

        return jax.vmap(_evaluate_single_)(evaluation_points)



    # def _interpolate_single(self, evaluation_points: jnp.ndarray) -> jnp.ndarray:
    #     # Compute array of differences (x - x_j)
    #     differences: jnp.ndarray = evaluation_points[:, None] - self._nodes_[None, :]
    #
    #     # Check if x exactly matches any interpolation node (True where difference is zero)
    #     exact_matches: jnp.ndarray = (differences == 0.0)
    #
    #     # Return the exact function value if x matches a node
    #     # Otherwise compute the barycentric interpolation using weights and differences
    #
    #     exact_match_indices: jnp.ndarray = jnp.argmax(exact_matches, axis=1)
    #     values_for_exact_matches: jnp.ndarray = self._values_[exact_match_indices]
    #
    #     addition_values: jnp.ndarray = self._weights_ / differences
    #
    #
    #     return jnp.where(
    #         jnp.any(exact_matches, axis=1),
    #         values_for_exact_matches,
    #         jnp.sum((self._weights_ * self._values_) / differences) / jnp.sum(self._weights_ / differences)
    #     )



    # def evaluate(self, x: jnp.ndarray) -> jnp.ndarray:
    #     return jax.jit(jax.vmap(lambda x_i: self._interpolate_single(x_i)))(x)
