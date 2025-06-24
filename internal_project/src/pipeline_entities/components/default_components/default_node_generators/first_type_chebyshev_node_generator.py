from functools import partial

import jax
from jax import numpy as jnp

from pipeline_entities.components.abstracts.node_generator import NodeGenerator
from pipeline_entities.components.decorators.pipeline_component import pipeline_component
from utils import jax_utils

# TODO: Update class!
# @pipeline_component(id="Chebyshev1")
# class FirstTypeChebyshevNodeGenerator(NodeGenerator):
#
#     @partial(jax.jit, static_argnums=0)
#     def generate_nodes(self) -> jnp.ndarray:
#         # Rescale int array for better stability
#         nodes: jnp.ndarray = jnp.arange(1, 2 * self.__node_count__ + 1, step=2, dtype=self.__data_type__)
#         nodes = jnp.multiply(nodes, jnp.pi / (2 * self.__node_count__))
#
#         nodes = jnp.cos(nodes)
#
#         jax_utils.rescale_array_to_interval(nodes, (-1, 1), self.__interval__)
#
#         # jnp.multiply(nodes, (self.__interval__[1] - self.__interval__[0]) / 2)
#         # jnp.add(nodes, (self.__interval__[0] + self.__interval__[1]) / 2)
#
#         return nodes #TODO rescaling does return an array!
#
#
#
#     def __repr__(self) -> str:
#         return "Node generator for type 1 chebyshev points"