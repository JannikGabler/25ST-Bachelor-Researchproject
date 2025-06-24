from functools import partial

import jax

from pipeline_entities.component_meta_info.default_component_meta_infos.node_generators.equidistant_node_generator_meta_info import \
    equidistant_node_generator_meta_info
from pipeline_entities.components.abstracts.node_generator import NodeGenerator
import jax.numpy as jnp

from pipeline_entities.components.decorators.pipeline_component import pipeline_component


@pipeline_component(id="equidistant node generator", type=NodeGenerator, meta_info=equidistant_node_generator_meta_info)
class EquidistantNodeGenerator(NodeGenerator):



    ######################
    ### Public methods ###
    ######################
    @partial(jax.jit, static_argnums=0)
    def perform_action(self) -> jnp.ndarray:
        return jnp.linspace(self.__interval__[0], self.__interval__[1], self.__node_count__, dtype=self.__data_type__) #TODO



    def __repr__(self) -> str:
        return "Node generator for equidistant nodes"