import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod

from exceptions.invalid_argument_exception import InvalidArgumentException


class Interpolant(ABC):
    ###############################
    ### Attributes of instances ###
    ###############################
    _required_evaluation_points_shape_: int | None
    _compiled_jax_callable_: callable | None



    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        self._evaluation_points_count_ = None
        self._compiled_jax_callable_ = None



    ######################
    ### Public methods ###
    ######################
    def recompile(self, amount_of_evaluation_points: int):
        dummy_array: jnp.ndarray = jnp.empty(amount_of_evaluation_points)

        self._compiled_jax_callable_ = jax.jit(self._internal_evaluate_).lower(dummy_array).compile()



    def evaluate(self, evaluation_points: jnp.ndarray):
        if evaluation_points.shape != self._required_evaluation_points_shape_:
            raise InvalidArgumentException(f"This Interpolant was compiled for the shape {self._required_evaluation_points_shape_} but the given evaluation_points array has the shape {evaluation_points.shape}.")

        return self._compiled_jax_callable_(evaluation_points)



    #######################
    ### Private methods ###
    #######################
    @abstractmethod
    def _internal_evaluate_(self, evaluation_points: jnp.ndarray):
        pass