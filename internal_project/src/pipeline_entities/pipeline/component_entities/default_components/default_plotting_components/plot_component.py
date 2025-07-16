import matplotlib.pyplot as plt

from pipeline_entities.pipeline.component_entities.component_meta_info.default_component_meta_infos.plot_components.plot_components import \
    plot_component_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore
import jax.numpy as jnp

from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData


@pipeline_component(id="interpolant plotter", type=InterpolationCore, meta_info=plot_component_meta_info)
class InterpolantPlotComponent(InterpolationCore):
    """
    TODO
    """

    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        pipeline_data: PipelineData = self._pipeline_data_[0]
        nodes: jnp.ndarray = pipeline_data.interpolation_nodes
        interpolation_interval: jnp.ndarray = pipeline_data.interpolation_interval
        
        # Erzeuge ein fein aufgelöstes Gitter für das Intervall:
        evaluation_points = jnp.linspace(interpolation_interval[0], interpolation_interval[1], 500)

        # Berechne Funktionswerte:
        true_function_values  = pipeline_data.function_callable(evaluation_points)

        # Berechne Interpolant-Werte:
        interpolated_values = pipeline_data.interpolant.evaluate(evaluation_points)

        plt.figure(figsize=(8, 5))
        plt.plot(evaluation_points, true_function_values, label="Original function")
        plt.plot(evaluation_points, interpolated_values, label="Interpolant")
        plt.scatter(nodes, pipeline_data.function_callable(nodes), color='red', label="Nodes")
        plt.legend()
        plt.title("Interpolant Plot")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()

        return pipeline_data



