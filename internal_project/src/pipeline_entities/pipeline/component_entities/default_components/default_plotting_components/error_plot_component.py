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
class ErrorPlotComponent(InterpolationCore):
    def perform_action(self) -> PipelineData:
        all_data = self._pipeline_data_
        reference_data = next((d for d in all_data if d.function_callable is not None), all_data[0])

        nodes = reference_data.interpolation_nodes
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

            for i, data in enumerate(all_data):
                interpolant = data.interpolant
                if interpolant is None:
                    continue

                compiled_interpolant: CompiledInterpolant = interpolant.compile(500, data.data_type)
                x_eval = jnp.linspace(interval[0], interval[1], 500).astype(compiled_interpolant.used_data_type)
                x_eval = x_eval.reshape(compiled_interpolant.required_evaluation_points_shape)

                raw_name = type(interpolant).__name__.replace("Interpolant", "")
                name = re.sub(r'(?<!^)(?=[A-Z])', ' ', raw_name)

                try:
                    y_interp = compiled_interpolant.evaluate(x_eval)
                except Exception:
                    print(f"[SKIP] {name}: Evaluation failed (exception).")
                    continue

                is_finite = jnp.isfinite(y_interp)
                if not jnp.any(is_finite):
                    print(f"[SKIP] {name}: All values are NaN or Inf.")
                    continue

                y_interp = y_interp[is_finite]
                x_eval_finite = x_eval[is_finite]
                y_ref = y_true[is_finite]

                abs_err = jnp.abs(y_ref - y_interp)

                if error_type == "relative":
                    err = abs_err / jnp.where(jnp.abs(y_ref) < EPSILON, EPSILON, jnp.abs(y_ref))
                else:
                    err = abs_err

                valid_mask = ~(jnp.isnan(err) | jnp.isinf(err))
                if not jnp.any(valid_mask):
                    print(f"[SKIP] {name}: All error values are NaN or Inf.")
                    continue

                plt.plot(
                    x_eval_finite[valid_mask], err[valid_mask],
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

            if ylog_value is True:
                plt.yscale("log")

            plt.show(block=False)

        return reference_data
