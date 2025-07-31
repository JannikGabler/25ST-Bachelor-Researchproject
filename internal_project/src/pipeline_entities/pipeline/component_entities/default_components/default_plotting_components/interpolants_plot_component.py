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
    COLORS = [
        "black",
        "#66c2a5",  # grünlich
        "#fc8d62",  # orange
        "#8da0cb",  # bläulich
        "#e78ac3",  # pink
        "#a6d854",  # hellgrün
        "#ffd92f",  # gelb
        "#e5c494",  # beige
        "#b3b3b3"  # grau
    ]
    LINESTYLES = [
        #'-',  # durchgezogen (solid)
        '--',  # gestrichelt (dashed)
        '-.',  # strich-punkt (dashdot)
        ':',  # gepunktet (dotted)
        (0, (1, 1)),  # sehr feine Punkte
        (0, (5, 5)),  # lange Striche mit Lücken
        (0, (3, 5, 1, 5)),  # Striche mit feinen Punkten
    ]

    AMOUNT_OF_EVALUATION_POINTS = 50


    def perform_action(self) -> PipelineData:
        #fig, ax = plt.subplots(figsize=(10, 6))
        plt.figure(figsize=(10, 6))

        self._plot_interpolation_points_(self._pipeline_data_[0].interpolation_nodes, self._pipeline_data_[0].interpolation_values)

        # TODO (multiple function callables)
        interval: jnp.ndarray = self._pipeline_data_[0].interpolation_interval
        original_function: callable = self._pipeline_data_[0].function_callable

        x_array: jnp.ndarray = jnp.linspace(interval[0], interval[1], self.AMOUNT_OF_EVALUATION_POINTS)

        original_function_values: jnp.ndarray = original_function(x_array)
        original_function_values = self._replace_nan_(original_function_values)

        y_limits: tuple[jnp.ndarray, jnp.ndarray] = self._calc_limits_(original_function_values)

        self._plot_original_function(x_array, original_function_values, y_limits)
        self._plot_interpolants_(x_array, y_limits)

        self._finish_up_plot(interval)

        return self._pipeline_data_[0]



    @classmethod
    def _plot_interpolation_points_(cls, interpolation_nodes: jnp.ndarray, interpolation_values: jnp.ndarray) -> None:
        size = cls._adaptive_scatter_size_(len(interpolation_nodes), modifier=0.7)
        plt.scatter(interpolation_nodes, interpolation_values, color='red', s=size, label="Interpolation Nodes", zorder=10)



    @staticmethod
    def _calc_limits_(original_function_values: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        valid_mask: jnp.ndarray = jnp.isfinite(original_function_values)

        clean_values = original_function_values[valid_mask]
        min_value = jnp.min(clean_values)
        max_value = jnp.max(clean_values)
        difference = 7 * (max_value - min_value)
        return min_value - difference, max_value + difference



    @classmethod
    def _plot_original_function(cls, x_array: jnp.ndarray, function_values: jnp.ndarray, y_limits: tuple[jnp.ndarray, jnp.ndarray]) -> None:
        x_intervals, y_intervals, inf_indices = cls._split_up_function_values_(x_array, function_values)

        for x_interval, y_interval in zip(x_intervals, y_intervals):
            plt.plot(x_interval, y_interval, label="Original Function", linewidth=4, color=cls.COLORS[0], alpha=0.6, zorder=1)

        for inf_index in inf_indices:
            value: jnp.ndarray = function_values[inf_index]
            clamped_value = y_limits[0] if jnp.isneginf(value) else y_limits[1]
            size = cls._adaptive_scatter_size_(len(x_array))
            plt.scatter(x_array[inf_index], clamped_value, color=cls.COLORS[0], s=size, zorder=10)



    def _plot_interpolants_(self, x_array: jnp.ndarray, y_limits: tuple[jnp.ndarray, jnp.ndarray]) -> None:

        for i, data in enumerate(self._pipeline_data_):
            if data.interpolant is not None:
                interpolant: CompiledInterpolant = data.interpolant.compile(len(x_array), data.data_type)
                function_values: jnp.ndarray = interpolant.evaluate(x_array.astype(data.data_type))
                function_values = self._replace_nan_(function_values)
                function_values = self._clamp_function_values_(function_values, y_limits)

                x_intervals, y_intervals, inf_indices = self._split_up_function_values_(x_array, function_values)

                plt.plot([], [], label=f"{data.interpolant.name}", color=self.COLORS[i+1])

                for j, (x_interval, y_interval) in enumerate(zip(x_intervals, y_intervals)):
                    plt.plot(x_interval, y_interval, linewidth=2, color=self.COLORS[i+1 % len(self.COLORS)],
                        linestyle=self.LINESTYLES[i % len(self.LINESTYLES)] ,zorder=2)

                for inf_index in inf_indices:
                    value: jnp.ndarray = function_values[inf_index]
                    clamped_value = y_limits[0] - i/4 if jnp.isneginf(value) else y_limits[1] + i/4
                    size=self._adaptive_scatter_size_(len(x_array))
                    plt.scatter(x_array[inf_index], clamped_value, color=self.COLORS[i+1], s=size, zorder=10)



    @staticmethod
    def _replace_nan_(array: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(jnp.isnan(array), jnp.inf, array)



    @classmethod
    def _split_up_function_values_(cls, x_array: jnp.ndarray, function_values: jnp.ndarray) -> tuple[list[jnp.ndarray], list[jnp.ndarray], list[int]]:
        indices_list, inf_indices = cls._get_indices_of_piecewise_intervals_(function_values)
        nodes_list = []
        values_list = []

        for indices in indices_list:
            nodes = x_array[indices[0]:indices[-1] + 1]
            nodes_list.append(nodes)

            values = function_values[indices[0]:indices[-1] + 1]
            values_list.append(values)

        return nodes_list, values_list, inf_indices


    @staticmethod
    def _get_indices_of_piecewise_intervals_(function_values: jnp.ndarray) -> tuple[list[list[int]], list[int]]:
        inf_indices = []

        indices_list = []
        indices = []

        for i, value in enumerate(function_values):
            if jnp.isinf(value) or jnp.isneginf(value):
                inf_indices.append(i)

                if len(indices) > 0:
                    indices_list.append(indices)

            else:
                indices.append(i)

        if len(indices) > 0:
            indices_list.append(indices)

        return indices_list, inf_indices



    @staticmethod
    def _clamp_function_values_(function_values: jnp.ndarray, y_limits: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        clamped_values = jnp.where(function_values < y_limits[0], -jnp.inf, function_values)
        clamped_values = jnp.where(clamped_values > y_limits[1], jnp.inf, clamped_values)
        return clamped_values



    @staticmethod
    def _finish_up_plot(interval: jnp.ndarray) -> None:
        plt.title(f"Original function and interpolants on [{interval[0]}, {interval[1]}]")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=False)



    @staticmethod
    def _adaptive_scatter_size_(num_points: int, modifier: float = 1.0, min_size: float = 10.0) -> float:
        """Berechnet eine adaptive Scatter-Größe, die mit Anzahl der Punkte abnimmt, aber nicht kleiner als min_size wird."""
        base_size = 200 * modifier
        size = base_size / (num_points ** 0.5)  # Quadratwurzel-Abnahme: visuell angenehm
        return max(size, min_size)


    # def perform_action(self) -> PipelineData:
    #     all_data = self._pipeline_data_
    #     reference_data = next((d for d in all_data if d.function_callable is not None), all_data[0])
    #
    #     nodes = reference_data.interpolation_nodes
    #     interval = reference_data.interpolation_interval
    #     f = reference_data.function_callable
    #
    #     x_plot = jnp.linspace(interval[0], interval[1], 500)
    #     y_true = f(x_plot) if f else None
    #
    #     plt.figure(figsize=(10, 6))
    #
    #     if y_true is not None:
    #         plt.plot(x_plot, y_true, '--', label="Original Function", linewidth=2.5, color='black', zorder=1)
    #
    #     if nodes is not None and f is not None:
    #         plt.scatter(nodes, f(nodes), color='red', s=50, label="Interpolation Nodes", zorder=10)
    #
    #     colors = ['blue', 'green', 'orange', 'purple', 'brown']
    #
    #     for i, data in enumerate(all_data):
    #         interpolant = data.interpolant
    #         if interpolant is None:
    #             continue
    #
    #         compiled_interpolant: CompiledInterpolant = interpolant.compile(500, data.data_type)
    #
    #         x_eval = jnp.linspace(interval[0], interval[1], 500).astype(compiled_interpolant.used_data_type)
    #         x_eval = x_eval.reshape(compiled_interpolant.required_evaluation_points_shape)
    #
    #         raw_name = type(interpolant).__name__.replace("Interpolant", "")
    #         name = re.sub(r'(?<!^)(?=[A-Z])', ' ', raw_name)
    #
    #         try:
    #             y_interp = compiled_interpolant.evaluate(x_eval)
    #         except Exception:
    #             print(f"[SKIP] {name}: Evaluation failed (exception).")
    #             continue
    #
    #         is_finite = jnp.isfinite(y_interp)
    #         if not jnp.any(is_finite):
    #             print(f"[SKIP] {name}: All values are NaN or Inf.")
    #             continue
    #
    #         x_plot_filtered = x_eval[is_finite]
    #         y_plot_filtered = y_interp[is_finite]
    #
    #         plt.plot(
    #             x_plot_filtered,
    #             y_plot_filtered,
    #             label=f"{name} Interpolant",
    #             color=colors[i % len(colors)],
    #             linestyle='-',
    #             linewidth=1.8
    #         )
    #
    #     if y_true is not None:
    #         max_value = float(jnp.max(y_true))
    #         min_value = float(jnp.min(y_true))
    #         difference = 7 * (max_value - min_value)
    #         plt.ylim(min_value - difference, max_value + difference)
    #     else:
    #         plt.ylim(-1, 1)
    #
    #     plt.title(f"Original Function and Interpolation Curves on [{interval[0]}, {interval[1]}]")
    #     plt.xlabel("x")
    #     plt.ylabel("f(x)")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()
    #     plt.close()
    #
    #     return reference_data