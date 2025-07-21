import matplotlib.pyplot as plt
import re

from interpolants.abstracts.compiled_interpolant import CompiledInterpolant
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.plot_components.interpolants_plot_component_meta_info import \
    plot_component_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore
import jax.numpy as jnp

from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData


@pipeline_component(id="interpolant plotter", type=InterpolationCore, meta_info=plot_component_meta_info)
class InterpolantsPlotComponent(InterpolationCore):
    """
    Creates plots for any number of interpolation methods.
    """

    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        all_data = self._pipeline_data_

        reference_data = next((d for d in all_data if d.function_callable is not None), all_data[0])

        nodes = reference_data.interpolation_nodes
        interval = reference_data.interpolation_interval
        f = reference_data.function_callable

        x_eval = jnp.linspace(interval[0], interval[1], 500)
        y_true = f(x_eval) if f else None

        plt.figure(figsize=(10, 6))

        if y_true is not None:
            plt.plot(x_eval, y_true, '--', label="Original Function", linewidth=2.5, color='black', zorder=1)

        if nodes is not None and f is not None:
            plt.scatter(nodes, f(nodes), color='red', s=50, label="Interpolation Nodes", zorder=10)

        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        # linestyles = ['-', '--', '-.', ':']

        for i, data in enumerate(all_data):
            interpolant = data.interpolant
            if interpolant is None:
                continue

            compiled_interpolant: CompiledInterpolant = interpolant.compile(len(x_eval), data.data_type)
            y_interp = compiled_interpolant.evaluate(x_eval)
            raw_name = type(interpolant).__name__.replace("Interpolant", "")
            name = re.sub(r'(?<!^)(?=[A-Z])', ' ', raw_name)

            plt.plot(
                x_eval,
                y_interp,
                label=f"{name} Interpolant",
                color=colors[i % len(colors)],
                linestyle='-',
                # linestyle=linestyles[i % len(linestyles)],
                linewidth=1.8
            )

        plt.title("Interpolants vs. Original Function")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return reference_data



