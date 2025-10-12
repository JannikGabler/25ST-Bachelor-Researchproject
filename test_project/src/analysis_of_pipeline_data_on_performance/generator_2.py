from functools import partial

import jax
import jax.numpy as jnp

from analysis_of_pipeline_data_on_performance.pipeline_data import PipelineData
from utils import jax_utils


class Generator2:
    __data__: PipelineData

    def __init__(self, data: PipelineData) -> None:
        self.__data__ = data

    @partial(jax.jit, static_argnums=0)
    def generate_nodes(self) -> jnp.ndarray:
        # Rescale int array for better stability
        nodes: jnp.ndarray = jnp.arange(
            1, 2 * self.__data__.node_count + 1, step=2, dtype=self.__data__.data_type
        )
        nodes = jnp.multiply(nodes, jnp.pi / (2 * self.__data__.node_count))

        nodes = jnp.cos(nodes)

        jax_utils.rescale_array_to_interval(
            nodes, (-1, 1), self.__data__.interpolation_interval
        )

        # jnp.multiply(nodes, (self.__interval__[1] - self.__interval__[0]) / 2)
        # jnp.add(nodes, (self.__interval__[0] + self.__interval__[1]) / 2)

        return nodes

    def __repr__(self) -> str:
        return "Node generator for type 1 chebyshev points"
