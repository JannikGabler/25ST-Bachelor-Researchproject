import matplotlib.pyplot as plt
import re

from interpolants.abstracts.compiled_interpolant import CompiledInterpolant
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.plot_components.interpolants_plot_component_meta_info import \
    interpolants_component_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore
import jax.numpy as jnp

from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData


@pipeline_component(id="interpolant plotter", type=InterpolationCore, meta_info=interpolants_component_meta_info)
class InterpolantsPlotComponent(InterpolationCore):
    def perform_action(self) -> PipelineData:
        all_data = self._pipeline_data_
        reference_data = next((d for d in all_data if d.function_callable is not None), all_data[0])

        nodes = reference_data.interpolation_nodes
        interval = reference_data.interpolation_interval
        f = reference_data.function_callable

        x_plot = jnp.linspace(interval[0], interval[1], 500)
        y_true = f(x_plot) if f else None

        plt.figure(figsize=(10, 6))

        if y_true is not None:
            plt.plot(x_plot, y_true, '--', label="Original Function", linewidth=2.5, color='black', zorder=1)

        if nodes is not None and f is not None:
            plt.scatter(nodes, f(nodes), color='red', s=50, label="Interpolation Nodes", zorder=10)

        colors = ['blue', 'green', 'orange', 'purple', 'brown']

        for i, data in enumerate(all_data):
            interpolant = data.interpolant
            if interpolant is None:
                continue

            compiled_interpolant: CompiledInterpolant = interpolant.compile(500, data.data_type)

            x_eval = jnp.linspace(interval[0], interval[1], 500).astype(compiled_interpolant.used_data_type)
            x_eval = x_eval.reshape(compiled_interpolant.required_evaluation_points_shape)

            try:
                y_interp = compiled_interpolant.evaluate(x_eval)
            except Exception as e:
                print(f"[ERROR] Evaluation failed for {type(interpolant).__name__} with dtype {x_eval.dtype}: {e}")
                continue

            is_finite = jnp.isfinite(y_interp)

            if not jnp.any(is_finite):
                print(f"[WARN] All values are NaN or Inf for {type(interpolant).__name__}, skipping plot.")
                print(f"First 10 computed y_interp values: {y_interp[:10]}")
                print(f"Indices with NaN or Inf: {jnp.where(~is_finite)[0]}")
                continue

            x_eval_valid = x_eval[is_finite]
            y_interp_valid = y_interp[is_finite]

            raw_name = type(interpolant).__name__.replace("Interpolant", "")
            name = re.sub(r'(?<!^)(?=[A-Z])', ' ', raw_name)

            plt.plot(
                x_eval_valid,
                y_interp_valid,
                label=f"{name} Interpolant",
                color=colors[i % len(colors)],
                linestyle='-',
                linewidth=1.8
            )

        if y_true is not None:
            max_value = float(jnp.max(y_true))
            min_value = float(jnp.min(y_true))
            difference = 7*(max_value - min_value)
            plt.ylim(min_value - difference, max_value + difference)
        else:
            plt.ylim(-1, 1)

        plt.title(f"Original Function and Interpolation Curves on [{interval[0]}, {interval[1]}]")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()

        return reference_data
