import matplotlib.pyplot as plt
import re

from interpolants.abstracts.compiled_interpolant import CompiledInterpolant
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.plot_components.error_plot_component_meta_info import \
    error_component_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore
import jax.numpy as jnp

from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData


@pipeline_component(id="error plotter", type=InterpolationCore, meta_info=error_component_meta_info)
class InterpolantsPlotComponent(InterpolationCore):
    def perform_action(self) -> PipelineData:
        all_data = self._pipeline_data_
        reference_data = next((d for d in all_data if d.function_callable is not None), all_data[0])

        interval = reference_data.interpolation_interval
        f = reference_data.function_callable

        if f is None:
            return reference_data

        x_base = jnp.linspace(interval[0], interval[1], 500)
        y_true = f(x_base)

        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        ylog_value = self._additional_execution_info_.overridden_attributes.get("ylogscale", None)
        EPSILON = 1e-12

        for error_type in ["absolute", "relative"]:
            plt.figure(figsize=(10, 6))
            has_non_positive = False

            for i, data in enumerate(all_data):
                interpolant = data.interpolant
                if interpolant is None:
                    continue

                compiled_interpolant: CompiledInterpolant = interpolant.compile(500, data.data_type)
                x_eval = jnp.linspace(interval[0], interval[1], 500).astype(compiled_interpolant.used_data_type)
                x_eval = x_eval.reshape(compiled_interpolant.required_evaluation_points_shape)

                try:
                    y_interp = compiled_interpolant.evaluate(x_eval)
                except Exception:
                    continue

                abs_err = jnp.abs(y_true - y_interp)

                if error_type == "relative":
                    err = abs_err / jnp.where(jnp.abs(y_true) < EPSILON, EPSILON, jnp.abs(y_true))
                else:
                    err = abs_err

                valid_mask = ~(jnp.isnan(err) | jnp.isinf(err))

                if not jnp.any(valid_mask):
                    continue

                if ylog_value is True and jnp.any(err[valid_mask] <= 0):
                    has_non_positive = True

                raw_name = type(interpolant).__name__.replace("Interpolant", "")
                name = re.sub(r'(?<!^)(?=[A-Z])', ' ', raw_name)

                plt.plot(
                    x_eval[valid_mask], err[valid_mask],
                    label=f"{name} Interpolant",
                    color=colors[i % len(colors)],
                    linewidth=1.8
                )

            title_map = {
                "absolute": "Absolute Errors of the Interpolants",
                "relative": "Relative Errors of the Interpolants"
            }
            ylabel_map = {
                "absolute": "Absolute Error |f(x) - p(x)|",
                "relative": "Relative Error |f(x) - p(x)| / |f(x)|"
            }

            plt.title(title_map[error_type])
            plt.xlabel("x")
            plt.ylabel(ylabel_map[error_type])
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            if self._additional_execution_info_.overridden_attributes["ylogscale"] is True:
                plt.yscale("log")

            plt.show()
            plt.close()

        return reference_data

