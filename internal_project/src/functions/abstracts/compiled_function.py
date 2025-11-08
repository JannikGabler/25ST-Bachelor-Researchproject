from jax.typing import DTypeLike

import jax.numpy as jnp

from exceptions.invalid_argument_exception import InvalidArgumentException


class CompiledFunction:
    """
    Wrapper for a compiled JAX callable with fixed input shape and dtype.
    """


    ###############################
    ### Attributes of instances ###
    ###############################
    _compiled_jax_callable_: callable
    _required_evaluation_points_shape_: tuple[int]
    _used_data_type_: DTypeLike


    ###################
    ### Constructor ###
    ###################
    def __init__(self, compiled_jax_callable: callable, required_evaluation_points_shape: tuple[int], used_data_type: DTypeLike) -> None:
        """
        Args:
            compiled_jax_callable: The compiled JAX function.
            required_evaluation_points_shape: Expected shape of the evaluation points.
            used_data_type: Expected data type of the evaluation points.
        """

        self._compiled_jax_callable_ = compiled_jax_callable
        self._required_evaluation_points_shape_ = required_evaluation_points_shape
        self._used_data_type_ = used_data_type


    ######################
    ### Public methods ###
    ######################
    def evaluate(self, evaluation_points: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the compiled function at the given points.

        Args:
            evaluation_points (jnp.ndarray): Input array of evaluation points.

        Returns:
            jnp.ndarray: Computed values of the compiled function.

        Raises:
            InvalidArgumentException: If shape or dtype of evaluation_points does not match the compiled specification.
        """

        if evaluation_points.shape != self._required_evaluation_points_shape_:
            raise InvalidArgumentException(
                f"This {repr(self.__class__.__name__)} was compiled for the shape "
                f"{self._required_evaluation_points_shape_} but the given evaluation_points array has the shape"
                f"{evaluation_points.shape}."
            )

        if evaluation_points.dtype != self._used_data_type_:
            raise InvalidArgumentException(
                f"This {repr(self.__class__.__name__)} was compiled for the data type "
                f"{repr(self._used_data_type_)} but the given evaluation_points array has the data type"
                f"{repr(evaluation_points.dtype)}."
            )

        result = self._compiled_jax_callable_(evaluation_points)

        return result


    #########################
    ### Getters & setters ###
    #########################
    @property
    def required_evaluation_points_shape(self) -> tuple[int]:
        """
        Get the required shape of the evaluation points.

        Returns:
            tuple[int]: Required evaluation points shape.
        """

        return self._required_evaluation_points_shape_


    @property
    def used_data_type(self) -> DTypeLike:
        """
        Get the expected dtype of the evaluation points.

        Returns:
            DTypeLike: Expected data type.
        """

        return self._used_data_type_


    ##########################
    ### Overridden methods ###
    ##########################
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(compiled_jax_callable={repr(self._compiled_jax_callable_)}, "
            f"required_evaluation_points_shape={repr(self._required_evaluation_points_shape_)}, "
            f"used_data_type={repr(self._used_data_type_)})"
        )


    def __str__(self) -> str:
        return self.__repr__()


    def __hash__(self) -> int:
        return hash((self._compiled_jax_callable_, self._required_evaluation_points_shape_, self._used_data_type_))


    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        else:
            return (self._compiled_jax_callable_ is other._compiled_jax_callable_ and self._required_evaluation_points_shape_
                    == other._required_evaluation_points_shape_ and self._used_data_type_ == other._used_data_type_)
