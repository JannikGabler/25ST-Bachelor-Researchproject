import jax
import jax.numpy as jnp
from numpy.typing import DTypeLike

from abc import ABC, abstractmethod

from functions.abstracts.compiled_function import CompiledFunction


class CompilableFunction(ABC):
    # TODO: outdated python doc
    """
    Represents an abstract base class designed to create interpolants that can be compiled
    for performance optimization. This class outlines a structure for defining interpolants
    that can support fast evaluation after JIT compilation. Users are expected to implement
    the necessary internal evaluation function(s).

    The class allows for:
      - Compiling interpolants for efficient evaluation over specified points.
      - Customizing evaluation behavior by implementing abstract methods.

    :ivar _data_type_: The data type used for calculations after compilation.
    :type _data_type_: DTypeLike
    """
    ###############################
    ### Attributes of instances ###
    ###############################
    _name_: str | None
    _data_type_: DTypeLike



    def __init__(self, name: str) -> None:
        self._name_ = name



    ######################
    ### Public methods ###
    ######################
    def compile(self, amount_of_evaluation_points: int, data_type: DTypeLike, **kwargs) -> CompiledFunction:
        shape: tuple[int] = (amount_of_evaluation_points,)
        self._data_type_ = data_type

        internal_evaluation_function: callable = self._get_internal_evaluate_function_(**kwargs)

        dummy_array: jnp.ndarray = jnp.empty(shape, dtype=data_type)
        compiled_jax_callable: callable = jax.jit(internal_evaluation_function).lower(dummy_array).compile()

        return CompiledFunction(compiled_jax_callable, shape, data_type)



    #########################
    ### Getters & setters ###
    #########################
    @property
    def name(self) -> str:
        return self._name_



    #######################
    ### Private methods ###
    #######################
    @abstractmethod
    def _get_internal_evaluate_function_(self, **kwargs) -> callable:
        pass
