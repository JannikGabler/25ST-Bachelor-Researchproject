import math
from functools import partial

import jax
import jax.numpy as jnp

from benchmark_aot_vs_jit.pipeline_data import PipelineData
from utils import jax_utils


class PurePythonGenerator:
    __data__: PipelineData

    def __init__(self, data: PipelineData) -> None:
        self.__data__ = data

    def generate_nodes(self) -> None:
        # Rescale int array for better stability
        nodes: list = [1 + 2 * i for i in range(self.__data__.node_count)]

        for i in range(len(nodes)):
            nodes[i] = math.cos(nodes[i] * math.pi / (2 * self.__data__.node_count))

        nodes: jnp.ndarray = jnp.array(nodes, dtype=self.__data__.data_type)

        self.__data__.nodes = jax_utils.rescale_array_to_interval(
            nodes,
            jnp.array([-1, 1], dtype=self.__data__.data_type),
            self.__data__.interpolation_interval,
        )

    def __repr__(self) -> str:
        return "Node generator for type 1 chebyshev points"
