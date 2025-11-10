import jax
import jax.numpy as jnp
from numpy.typing import DTypeLike

from abc import ABC, abstractmethod

from functions.abstracts.compiled_function import CompiledFunction


class CompilableFunction(ABC):
    """
    Abstract base class for functions that can be compiled with JAX for efficient evaluation.
    """


    ###############################
    ### Attributes of instances ###
    ###############################
    _name_: str | None
    _data_type_: DTypeLike


    def __init__(self, name: str) -> None:
        """
        Args:
            name: Name of the function.
        """

        self._name_ = name


    ######################
    ### Public methods ###
    ######################
    def compile(self, amount_of_evaluation_points: int, data_type: DTypeLike, **kwargs) -> CompiledFunction:
        """
        Compile the function for efficient evaluation on JAX arrays.

        Args:
            amount_of_evaluation_points (int): Number of evaluation points.
            data_type (DTypeLike): Data type of the evaluation arrays.
            **kwargs: Additional arguments passed to the internal evaluation function.

        Returns:
            CompiledFunction: A compiled function ready for fast evaluation.
        """

        shape: tuple[int] = (amount_of_evaluation_points,)
        self._data_type_ = data_type

        internal_evaluation_function: callable = self._get_internal_evaluate_function_(**kwargs)

        dummy_array: jnp.ndarray = jnp.empty(shape, dtype=data_type)
        compiled_jax_callable: callable = (jax.jit(internal_evaluation_function).lower(dummy_array).compile())

        return CompiledFunction(compiled_jax_callable, shape, data_type)


    #########################
    ### Getters & setters ###
    #########################
    @property
    def name(self) -> str:
        """
        Return the name of the function.

        Returns:
            str: the name of the compilable function.
        """
        return self._name_


    #######################
    ### Private methods ###
    #######################
    @abstractmethod
    def _get_internal_evaluate_function_(self, **kwargs) -> callable:
        pass
