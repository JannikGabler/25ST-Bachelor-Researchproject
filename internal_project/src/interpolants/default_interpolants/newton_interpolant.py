from typing import Any

from exceptions.invalid_argument_exception import InvalidArgumentException
from interpolants.abstracts.interpolant import Interpolant
import jax
import jax.numpy as jnp


class NewtonInterpolant(Interpolant):
    ###############################
    ### Attributes of instances ###
    ###############################
    _divided_differences_: jnp.ndarray
    _nodes_: jnp.ndarray



    ###################
    ### Constructor ###
    ###################
    def __init__(self, nodes: jnp.ndarray, divided_differences: jnp.ndarray):
        super().__init__()

        if nodes.shape != divided_differences.shape:
            raise InvalidArgumentException("The shapes of the given nodes and divided_differences arrays differ, although"
               f"they're required to be equal (shape of nodes: {nodes.shape}, shape of divided_differences:"
               f"{divided_differences.shape}).")

        self._nodes_ = nodes
        self._divided_differences_ = divided_differences



    ##########################
    ### Overridden methods ###
    ##########################
    def _get_internal_evaluate_function_(self) -> callable:
        if self._is_data_type_overridden_:
            return self._internal_evaluate_with_data_type_overriding
        else:
            return self._internal_evaluate_without_data_type_overriding



    def _is_data_type_overriding_required_(self) -> bool:
        return self._nodes_.dtype != self._divided_differences_.dtype

    def _get_data_type_for_no_overriding_(self) -> jnp.dtype:
        return self._nodes_.dtype



    def __repr__(self) -> str:
        return f"NewtonInterpolant(divided_differences={self._divided_differences_}, nodes={self._nodes_})"

    def __str__(self) -> str:
        return self.__repr__()



    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NewtonInterpolant):
            return False
        else:
            return (jnp.array_equal(self._divided_differences_, other._divided_differences_).item()
                    and jnp.array_equal(self._nodes_, other._nodes_).item())



    #######################
    ### Private methods ###
    #######################
    def _internal_evaluate_without_data_type_overriding(self, evaluation_points: jnp.ndarray) -> jnp.ndarray:
        n = self._divided_differences_.size
        initial_accumulator = jnp.zeros_like(evaluation_points)

        # Inner loop function implementing the nested form of the Newton polynomial, which corresponds to Horner's scheme for the Newton form
        def horner_step(i, val):
            reverse_index = n - 1 - i
            return val * (evaluation_points - self._nodes_[reverse_index]) + self._divided_differences_[reverse_index]

        return jax.lax.fori_loop(0, n, horner_step, initial_accumulator)



    def _internal_evaluate_with_data_type_overriding(self, evaluation_points: jnp.ndarray) -> jnp.ndarray:
        n = self._divided_differences_.size
        initial_accumulator: jnp.ndarray = jnp.zeros_like(evaluation_points, dtype=self._required_data_type_)
        nodes: jnp.ndarray = self._nodes_.astype(self._required_data_type_)
        divided_differences: jnp.ndarray = self._divided_differences_.astype(self._required_data_type_)

        # Inner loop function implementing the nested form of the Newton polynomial, which corresponds to Horner's scheme for the Newton form
        def horner_step(i, val):
            reverse_index = n - 1 - i
            return val * (evaluation_points - nodes[reverse_index]) + divided_differences[reverse_index]

        return jax.lax.fori_loop(0, n, horner_step, initial_accumulator)




