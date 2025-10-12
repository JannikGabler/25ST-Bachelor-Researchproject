from functools import partial

import jax
import jax.numpy as jnp

from benchmark_aot_vs_jit.pipeline_data import PipelineData
from utils import jax_utils


class JitGenerator:
    __data__: PipelineData

    def __init__(self, data: PipelineData) -> None:
        self.__data__ = data

    def generate_nodes(self) -> None:
        self.__data__.nodes = self.__generate_nodes__()

    @partial(jax.jit, static_argnums=0)
    def __generate_nodes__(self) -> jnp.ndarray:
        # Rescale int array for better stability
        nodes: jnp.ndarray = jnp.arange(
            1, 2 * self.__data__.node_count + 1, step=2, dtype=self.__data__.data_type
        )
        nodes = jnp.multiply(nodes, jnp.pi / (2 * self.__data__.node_count))
        nodes = jnp.cos(nodes)

        return jax_utils.rescale_array_to_interval(
            nodes,
            jnp.array([-1, 1], dtype=self.__data__.data_type),
            self.__data__.interpolation_interval,
        )

    def __repr__(self) -> str:
        return "Node generator for type 1 chebyshev points"
