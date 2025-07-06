import matplotlib.pyplot as plt

from pipeline_entities.component_meta_info.default_component_meta_infos.plot_components.plot_components import \
    plot_component_meta_info
from pipeline_entities.components.abstracts.interpolation_core import InterpolationCore
import jax.numpy as jnp

from pipeline_entities.components.decorators.pipeline_component import pipeline_component
from pipeline_entities.data_transfer.pipeline_data import PipelineData


@pipeline_component(id="interpolant plotter", type=InterpolationCore, meta_info=plot_component_meta_info)
class InterpolantPlotComponent(InterpolationCore):
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
        linestyles = ['-', '--', '-.', ':']

        for i, data in enumerate(all_data):
            interpolant = data.interpolant
            if interpolant is None:
                continue

            y_interp = interpolant.evaluate(x_eval)
            name = type(interpolant).__name__.replace("Interpolant", "")

            plt.plot(
                x_eval,
                y_interp,
                label=f"{name} Interpolant",
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2
            )

        plt.title("Interpolants vs. Original Function")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return reference_data



