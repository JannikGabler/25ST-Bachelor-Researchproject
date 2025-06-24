from functools import partial

import jax
import jax.numpy as jnp

from utils import jax_utils


class Generator1:
    __node_count__: int
    __data_type__: type
    __interval__: tuple[float, float]

    def __init__(self, node_count, data_type, interval) -> None:
        self.__node_count__ = node_count
        self.__data_type__ = data_type
        self.__interval__ = interval

    @partial(jax.jit, static_argnums=0)
    def generate_nodes(self) -> jnp.ndarray:
        # Rescale int array for better stability
        nodes: jnp.ndarray = jnp.arange(1, 2 * self.__node_count__ + 1, step=2, dtype=self.__data_type__)
        nodes = jnp.multiply(nodes, jnp.pi / (2 * self.__node_count__))

        nodes = jnp.cos(nodes)

        jax_utils.rescale_array_to_interval(nodes, (-1, 1), self.__interval__)

        # jnp.multiply(nodes, (self.__interval__[1] - self.__interval__[0]) / 2)
        # jnp.add(nodes, (self.__interval__[0] + self.__interval__[1]) / 2)

        return nodes



    def __repr__(self) -> str:
        return "Node generator for type 1 chebyshev points"