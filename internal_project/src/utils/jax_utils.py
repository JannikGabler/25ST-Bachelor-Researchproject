from typing import Tuple

import jax
import jax.numpy as jnp

from exceptions.not_instantiable_error import NotInstantiableError


class JaxUtils:
    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        raise NotInstantiableError("The class 'JaxUtils' can not be instantiated.")



    ######################
    ### Public methods ###
    ######################
    @staticmethod
    @jax.jit
    def rescale_array_to_interval(array: jnp.ndarray, old_interval: jnp.ndarray, new_interval: jnp.ndarray) -> jnp.ndarray:
        old_length = old_interval[1] - old_interval[0]
        new_length = new_interval[1] - new_interval[0]
        length_ratio = new_length / old_length

        array = jnp.multiply(array, length_ratio)
        return jnp.add(array, new_interval[0] - old_interval[0] * length_ratio)



    @staticmethod
    def relative_tolerances(array_1: jnp.ndarray, array_2: jnp.ndarray) -> jnp.ndarray:
        distances = jnp.abs(array_1 - array_2)
        minima = jnp.where(abs(array_1) < abs(array_2), abs(array_1), abs(array_2))
        return jnp.where(minima > 0, distances / minima, 0)


    @staticmethod
    def all_close_enough(array_1: jnp.ndarray, array_2: jnp.ndarray, rtol: float) -> bool:
        rtol_array = JaxUtils.relative_tolerances(array_1, array_2)
        return jnp.all(rtol_array <= rtol).item()








