from typing import Callable

import jax
import matplotlib.pyplot as plt

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

        self.interpolation_interval = pipeline_data.interpolation_interval
        self.function_callable = pipeline_data.function_callable
        self.nodes = pipeline_data.nodes
        self.interpolant = pipeline_data.interpolant



    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> None:
        # Erzeuge ein fein aufgelöstes Gitter für das Intervall:
        evaluation_points = jnp.linspace(self.interpolation_interval[0], self.interpolation_interval[1], 500)

        # Berechne Funktionswerte:
        true_function_values  = self.function_callable(evaluation_points)

        # Berechne Interpolant-Werte:
        interpolated_values = self.interpolant.evaluate(evaluation_points)

        plt.figure(figsize=(8, 5))
        plt.plot(evaluation_points, true_function_values, label="Original function")
        plt.plot(evaluation_points, interpolated_values, label="Interpolant")
        plt.scatter(self.nodes, self.function_callable(self.nodes), color='red', label="Nodes")
        plt.legend()
        plt.title("Interpolant Plot")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()



