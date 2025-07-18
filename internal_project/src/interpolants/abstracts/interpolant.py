from typing import Callable

import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod

from exceptions.invalid_argument_exception import InvalidArgumentException


class Interpolant(ABC):
    ###############################
    ### Attributes of instances ###
    ###############################
    _required_evaluation_points_shape_: tuple[int] | None
    _required_data_type_: jnp.dtype | None
    _is_data_type_overridden_: bool | None
    _compiled_jax_callable_: Callable | None



    ###################
    ### Constructor ###
    ###################
    def __init__(self) -> None:
        self._required_evaluation_points_shape_ = None
        self._required_data_type_ = None
        self._compiled_jax_callable_ = None



    ######################
    ### Public methods ###
    ######################
    def recompile(self, amount_of_evaluation_points: int, overwritten_data_type: jnp.dtype | None = None) -> None:
        self._required_evaluation_points_shape_ = (amount_of_evaluation_points,)
        self._required_data_type_ = self._calc_required_data_type_(overwritten_data_type)
        self._is_data_type_overridden_ = overwritten_data_type is not None

        evaluate_callable: callable = self._get_internal_evaluate_function_()

        dummy_array: jnp.ndarray = jnp.empty(amount_of_evaluation_points, dtype=self._required_data_type_)
        self._compiled_jax_callable_ = jax.jit(evaluate_callable).lower(dummy_array).compile()



    def evaluate(self, evaluation_points: jnp.ndarray) -> jnp.ndarray:
        if evaluation_points.shape != self._required_evaluation_points_shape_:
            raise InvalidArgumentException(f"This Interpolant was compiled for the shape {self._required_evaluation_points_shape_}"
                                           f"but the given evaluation_points array has the shape {evaluation_points.shape}.")
        if evaluation_points.dtype != self._required_data_type_:
            raise InvalidArgumentException(f"This Interpolant was compiled for the data type {self._required_data_type_}"
                                           f"but the given evaluation_points array has the data type {evaluation_points.dtype}.")

        return self._compiled_jax_callable_(evaluation_points)



    #######################
    ### Private methods ###
    #######################
    @abstractmethod
    def _get_internal_evaluate_function_(self) -> callable:
        pass


    @abstractmethod
    def _is_data_type_overriding_required_(self) -> bool:
        pass


    @abstractmethod
    def _get_data_type_for_no_overriding_(self) -> jnp.dtype:
        pass



    def _calc_required_data_type_(self, overwritten_data_type: jnp.dtype | None) -> jnp.dtype:
        if self._is_data_type_overriding_required_():
            return overwritten_data_type
        else:
            return self._get_data_type_for_no_overriding_()



    # @abstractmethod
    # def _internal_evaluate_(self, evaluation_points: jnp.ndarray):
    #     pass