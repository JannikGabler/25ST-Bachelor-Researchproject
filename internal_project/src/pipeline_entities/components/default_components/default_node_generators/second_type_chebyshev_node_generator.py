from functools import partial

import jax

from jax import numpy as jnp
from pipeline_entities.components.abstracts.node_generator import NodeGenerator
from pipeline_entities.components.decorators.pipeline_component import pipeline_component
from utils import jax_utils


# TODO: Update class (AOT compilation, change generate_nodes to perform_action, ...
# @pipeline_component(id="Chebyshev2")
# class SecondTypeChebyshevNodeGenerator(NodeGenerator):
#
#     @partial(jax.jit, static_argnums=0)
#     def generate_nodes(self) -> jnp.ndarray:
#         # Rescale int array for better stability
#         nodes: jnp.ndarray = jnp.arange(0, self.__node_count__, dtype=self.__data_type__)
#         jnp.multiply(nodes, jnp.pi / self.__node_count__)
#
#         jnp.cos(nodes)
#
#         jax_utils.rescale_array_to_interval(nodes, (-1, 1), self.__interval__)
#
#         return nodes #TODO rescaling does return an array!
#
#
#
#     def __repr__(self) -> str:
#         return "Node generator for type 2 chebyshev points"