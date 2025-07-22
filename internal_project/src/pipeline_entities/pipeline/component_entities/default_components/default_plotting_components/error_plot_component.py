import matplotlib.pyplot as plt
import re

from interpolants.abstracts.compiled_interpolant import CompiledInterpolant
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.plot_components.interpolants_plot_component_meta_info import \
    plot_component_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore
import jax.numpy as jnp

from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData


@pipeline_component(id="error plotter", type=InterpolationCore, meta_info=plot_component_meta_info)
class InterpolantsPlotComponent(InterpolationCore):
    """
    Plots absolute and relative interpolation errors for all interpolants.
    """

    def perform_action(self) -> PipelineData:
        all_data = self._pipeline_data_

        reference_data = next((d for d in all_data if d.function_callable is not None), all_data[0])

        interval = reference_data.interpolation_interval
        f = reference_data.function_callable

        x_eval = jnp.linspace(interval[0], interval[1], 500)
        y_true = f(x_eval) if f else None

        colors = ['blue', 'green', 'orange', 'purple', 'brown']

        # Absolute Error Plots
        plt.figure(figsize=(10, 6))
        for i, data in enumerate(all_data):
            interpolant = data.interpolant
            if interpolant is None or y_true is None:
                continue

            compiled_interpolant: CompiledInterpolant = interpolant.compile(len(x_eval), data.data_type)
            y_interp = compiled_interpolant.evaluate(x_eval)

            abs_err = jnp.abs(y_true - y_interp)

            raw_name = type(interpolant).__name__.replace("Interpolant", "")
            name = re.sub(r'(?<!^)(?=[A-Z])', ' ', raw_name)

            plt.plot(
                x_eval,
                abs_err,
                label=f"{name} Interpolant",
                color=colors[i % len(colors)],
                linestyle='-',
                linewidth=1.8
            )

        plt.title("Absolute Errors of the Interpolants")
        plt.xlabel("x")
        plt.ylabel("Absolute Error |f(x) - p(x)|")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()

        # Relative Error Plots
        plt.figure(figsize=(10, 6))
        for i, data in enumerate(all_data):
            interpolant = data.interpolant
            if interpolant is None or y_true is None:
                continue

            compiled_interpolant: CompiledInterpolant = interpolant.compile(len(x_eval), data.data_type)
            y_interp = compiled_interpolant.evaluate(x_eval)

            abs_err = jnp.abs(y_true - y_interp)
            rel_err = abs_err / jnp.where(jnp.abs(y_true) < 1e-12, 1e-12, jnp.abs(y_true))

            raw_name = type(interpolant).__name__.replace("Interpolant", "")
            name = re.sub(r'(?<!^)(?=[A-Z])', ' ', raw_name)

            plt.plot(
                x_eval,
                rel_err,
                label=f"{name} Interpolant",
                color=colors[i % len(colors)],
                linestyle='-',
                linewidth=1.8
            )

        plt.title("Relative Errors of the Interpolants")
        plt.xlabel("x")
        plt.ylabel("Relative Error |f(x) - p(x)| / |f(x)|")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()

        return reference_data
