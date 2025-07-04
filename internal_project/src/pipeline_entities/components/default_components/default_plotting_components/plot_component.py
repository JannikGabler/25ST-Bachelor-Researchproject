from typing import Callable

import jax

from data_structures.interpolants.abstracts.interpolant import Interpolant
from pipeline_entities.component_meta_info.default_component_meta_infos.plot_components.plot_components import \
    plot_component_meta_info
from pipeline_entities.components.abstracts.interpolation_core import InterpolationCore
import jax.numpy as jnp

from pipeline_entities.components.decorators.pipeline_component import pipeline_component
from pipeline_entities.data_transfer.pipeline_data import PipelineData


@pipeline_component(id="Interpolant Plotter", type=InterpolationCore, meta_info=plot_component_meta_info)
class EquidistantNodeGenerator(InterpolationCore):
    """
    TODO
    """
    ###############################
    ### Attributes of instances ###
    ###############################
    _compiled_jax_callable_: callable



    ###################
    ### Constructor ###
    ###################
    def __init__(self, pipeline_data: PipelineData) -> None:
        super().__init__(pipeline_data)

        interpolation_interval = pipeline_data.interpolation_interval
        function_callable = pipeline_data.function_callable
        nodes = pipeline_data.nodes
        interpolant = pipeline_data.interpolant


        self._compiled_jax_callable_ = self._create_compiled_callable_(nodes)




    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> None:
        interpolant = self._compiled_jax_callable_()
        self._pipeline_data_.interpolant = interpolant



    #######################
    ### Private methods ###
    #######################
    def _create_compiled_callable_(self, interpolation_interval: jnp.ndarray, function_callable:Callable[[jnp.ndarray], jnp.ndarray], nodes: jnp.ndarray, interpolant: Interpolant):

        def _internal_perform_action_() -> jnp.ndarray:
          "TODO"

        return (
            jax.jit(_internal_perform_action_)       # → XLA-compatible HLO
                .lower()    # → Low-Level-IR
                .compile()  # → executable Binary
        )