from typing import Tuple

import jax
import jax.numpy as jnp

from exceptions.invalid_argument_exception import InvalidArgumentException
from exceptions.not_instantiable_error import NotInstantiableError


class JaxUtils:
    """
    Utility helpers for array operations using JAX. This class is not meant to be instantiated.
    """

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        """
        Raises:
            NotInstantiableError: Always raised when initialized to indicate that this class is not meant to be instantiated.
        """

        raise NotInstantiableError(f"The class {repr(self.__class__.__name__)} can not be instantiated.")


    ######################
    ### Public methods ###
    ######################
    @staticmethod
    @jax.jit
    def rescale_array_to_interval(array: jnp.ndarray, old_interval: jnp.ndarray, new_interval: jnp.ndarray) -> jnp.ndarray:
        """
        Linearly rescale all values in array from the old interval to the new interval.

        Args:
            array (jnp.ndarray): Input values to be rescaled.
            old_interval (jnp.ndarray): Source interval.
            new_interval (jnp.ndarray): Target interval.

        Returns:
            jnp.ndarray: Rescaled array.
        """

        old_length = old_interval[1] - old_interval[0]
        new_length = new_interval[1] - new_interval[0]
        length_ratio = new_length / old_length

        array = jnp.multiply(array, length_ratio)
        return jnp.add(array, new_interval[0] - old_interval[0] * length_ratio)


    @staticmethod
    def relative_tolerances(array_1: jnp.ndarray, array_2: jnp.ndarray) -> jnp.ndarray:
        """
        Compute element-wise relative differences between two arrays.

        Args:
            array_1 (jnp.ndarray): First array.
            array_2 (jnp.ndarray): Second array.

        Returns:
            jnp.ndarray: Array of relative differences.
        """

        distances = jnp.abs(array_1 - array_2)
        minima = jnp.where(abs(array_1) < abs(array_2), abs(array_1), abs(array_2))
        return jnp.where(minima > 0, distances / minima, 0)


    @staticmethod
    def absolute_tolerances(array_1: jnp.ndarray, array_2: jnp.ndarray) -> jnp.ndarray:
       """
       Compute element-wise absolute differences between two arrays.

       Args:
           array_1 (jnp.ndarray): First array.
           array_2 (jnp.ndarray): Second array.

       Returns:
           jnp.ndarray: Array of absolute differences.
       """

       return jnp.abs(array_1 - array_2)


    @staticmethod
    def all_close_enough(array_1: jnp.ndarray, array_2: jnp.ndarray, atol: float | None = None, rtol: float | None = None) -> bool:
        """
        Check if two arrays are element-wise within absolute and/or relative tolerances.

        Args:
            array_1 (jnp.ndarray): First array.
            array_2 (jnp.ndarray): Second array.
            atol (float | None): Absolute tolerance threshold.
            rtol (float | None): Relative tolerance threshold.

        Returns:
            bool: True if all elements satisfy at least one enabled tolerance check, False otherwise.

        Raises:
            InvalidArgumentException: If both the absolute and relative tolerance threshold are None.
        """

        if atol is None and rtol is None:
            raise InvalidArgumentException("Either the absolut or relative tolerance must be specified.")

        result_array = jnp.zeros(array_1.shape, dtype=bool)

        if atol is not None:
            atol_array: jnp.ndarray = JaxUtils.absolute_tolerances(array_1, array_2)
            atol_satisfied_array: jnp.ndarray = (atol_array <= atol)
            result_array = jnp.logical_or(result_array, atol_satisfied_array)

        if rtol is not None:
            rtol_array: jnp.ndarray = JaxUtils.relative_tolerances(array_1, array_2)
            rtol_satisfied_array = (rtol_array <= rtol)
            result_array = jnp.logical_or(result_array, rtol_satisfied_array)

        return jnp.all(result_array).item()


