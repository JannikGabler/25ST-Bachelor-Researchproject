import math
from dataclasses import dataclass
from jax.typing import DTypeLike

import jax.numpy as jnp
from matplotlib.lines import Line2D

from constants.internal_logic_constants import InterpolantsPlotComponentConstants
from data_classes.pipeline_data.pipeline_data import PipelineData
from data_classes.plot_template.plot_template import PlotTemplate
from functions.abstracts.compilable_function import CompilableFunction
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import \
    AdditionalComponentExecutionData
from utils.plot_utils import PlotUtils


class InterpolantPlotComponentUtils:
    @dataclass
    class _Data_:
        amount_of_functions_to_plot: int | None = None
        functions: list[CompilableFunction] | None = None
        data_types: list[DTypeLike] | None = None
        evaluation_points: jnp.ndarray | None = None
        function_values: jnp.ndarray | None = None
        function_names: list[str] | None = None
        y_limits: jnp.ndarray | None = None
        scatter_size: float | None = None
        scatter_x_distance: jnp.ndarray | None = None
        scatter_y_distance: jnp.ndarray | None = None
        borders: jnp.ndarray | None = None
        connectable_segments: list[tuple[list[list[int]], list[list[int]]]] | None = None



    # TODO: __init__



    @classmethod
    def plot_data(cls, pipeline_data: list[PipelineData], additional_data: AdditionalComponentExecutionData) -> PlotTemplate:
        pd: PipelineData = pipeline_data[0]
        data: cls._Data_ = cls._Data_()
        template: PlotTemplate = PlotTemplate(figsize=InterpolantsPlotComponentConstants.FIGURE_SIZE)

        data.amount_of_functions_to_plot = len(pipeline_data) + 1
        data.evaluation_points = PlotUtils.create_plot_points(pd.interpolation_interval,
                                                              InterpolantsPlotComponentConstants.AMOUNT_OF_EVALUATION_POINTS)
        data.functions = [pd.original_function] + [pd.interpolant for pd in pipeline_data]
        data.data_types = [pd.data_type] + [_pd_.data_type for _pd_ in pipeline_data]
        data.function_values = cls._calc_function_values_(data, pipeline_data)
        data.function_names = [pd.original_function.name] + [pd.interpolant.name for pd in pipeline_data]
        data.y_limits = cls._calc_y_limits_(data.function_values[0])
        cls._clean_up_function_values_(data)
        data.scatter_size = PlotUtils.scatter_size_for_equidistant_circles(
            InterpolantsPlotComponentConstants.AMOUNT_OF_INF_SCATTER_POINTS, InterpolantsPlotComponentConstants.FIGURE_SIZE[0])
        data.scatter_x_distance = PlotUtils.scatter_distance_for_equidistant_circles(
            InterpolantsPlotComponentConstants.AMOUNT_OF_INF_SCATTER_POINTS, pd.interpolation_interval[1] - pd.interpolation_interval[0])
        data.scatter_y_distance = cls._calc_scatter_y_distance_(template, data)
        data.borders = cls._calc_borders_(data, pd.interpolation_interval)

        data.connectable_segments = []
        for i, function_values in enumerate(data.function_values):
            segments_to_plot, segments_to_scatter = cls._extract_indices_of_connectable_segments_(function_values)
            data.connectable_segments.append((segments_to_plot, segments_to_scatter))

        cls._plot_interpolation_points_(template, pd)
        cls._plot_segments_(template, data)

        template.ax.scatter(data.evaluation_points[0], data.borders[1, 0], alpha=0)
        template.ax.scatter(data.evaluation_points[0], data.borders[1, 1], alpha=0)

        meta_info_str: str = PlotUtils.create_plot_meta_info_str(pipeline_data, additional_data)

        template.fig.suptitle(f"Interpolant plot")
        template.ax.set_title(meta_info_str, fontsize=10)
        template.ax.set_xlabel("x")
        template.ax.set_ylabel("y")

        cls._set_legend_(template, data)
        template.ax.grid()
        template.fig.tight_layout()

        return template



    @staticmethod
    def _calc_function_values_(data: _Data_, pipeline_data: list[PipelineData]) -> jnp.ndarray:
        function_values: jnp.ndarray = jnp.empty((data.amount_of_functions_to_plot, InterpolantsPlotComponentConstants.AMOUNT_OF_EVALUATION_POINTS))

        v: jnp.ndarray = PlotUtils.evaluate_function(pipeline_data[0].original_function, pipeline_data[0].data_type,
                                                     data.evaluation_points)
        function_values = function_values.at[0].set(v.astype(jnp.float32))

        for i, pd in enumerate(pipeline_data):
            v: jnp.ndarray = PlotUtils.evaluate_function(pd.interpolant, pd.data_type, data.evaluation_points)
            function_values = function_values.at[i + 1].set(v.astype(jnp.float32))

        return function_values



    @staticmethod
    def _calc_y_limits_(original_function_values: jnp.ndarray) -> jnp.ndarray:
        valid_mask: jnp.ndarray = jnp.isfinite(original_function_values)

        clean_values = original_function_values[valid_mask]
        min_value = jnp.min(clean_values)
        max_value = jnp.max(clean_values)
        difference = InterpolantsPlotComponentConstants.Y_LIMIT_FACTOR * (max_value - min_value)
        return jnp.array([min_value - difference, max_value + difference])



    @staticmethod
    def _clean_up_function_values_(data: _Data_) -> None:
        for i, function_values in enumerate(data.function_values):
            cleaned_values = PlotUtils.replace_nan_with_inf(function_values)
            cleaned_values = PlotUtils.clamp_function_values(cleaned_values, data.y_limits)
            data.function_values = data.function_values.at[i].set(cleaned_values)



    @staticmethod
    def _calc_scatter_y_distance_(template: PlotTemplate, data: _Data_) -> jnp.ndarray:
        scatter_y_distance_inch: float = math.sqrt(data.scatter_size / math.pi) / 72
        height_in_inches: float = InterpolantsPlotComponentConstants.FIGURE_SIZE[1] / template.fig.dpi

        numerator = data.y_limits[1] - data.y_limits[0]
        denominator = 100 * height_in_inches / scatter_y_distance_inch - 2 * data.amount_of_functions_to_plot
        return 3 * numerator / denominator



    @staticmethod
    def _calc_borders_(data: _Data_, interval: jnp.ndarray) -> jnp.ndarray:
        borders: jnp.ndarray = jnp.empty((2, 2))

        borders = borders.at[0].set(interval)
        borders = borders.at[1, 0].set(data.y_limits[0] - data.amount_of_functions_to_plot * data.scatter_y_distance)
        borders = borders.at[1, 1].set(data.y_limits[1] + data.amount_of_functions_to_plot * data.scatter_y_distance)
        return borders



    @staticmethod
    def _extract_indices_of_connectable_segments_(function_values: jnp.ndarray) -> tuple[list[list[int]], list[list[int]]]:
        segments_to_plot: list[list[int]] = []
        segments_to_scatter: list[list[int]] = []

        current_plot_segment: list[int] = []
        current_scatter_segment: list[int] = []

        for i, value in enumerate(function_values):
            if jnp.isfinite(value):
                current_plot_segment.append(i)

                if len(current_scatter_segment) > 0:
                    segments_to_scatter.append(current_scatter_segment)
                    current_scatter_segment = []
            else:
                current_scatter_segment.append(i)

                if len(current_plot_segment) > 0:
                    segments_to_plot.append(current_plot_segment)
                    current_plot_segment = []

        if len(current_scatter_segment) > 0:
            segments_to_scatter.append(current_scatter_segment)
        if len(current_plot_segment) > 0:
            segments_to_plot.append(current_plot_segment)

        return segments_to_plot, segments_to_scatter



    @staticmethod
    def _plot_interpolation_points_(template: PlotTemplate, pipeline_data: PipelineData) -> None:
        interpolation_nodes: jnp.ndarray = pipeline_data.interpolation_nodes
        interpolation_values: jnp.ndarray = pipeline_data.interpolation_values

        size = PlotUtils.adaptive_scatter_size(len(interpolation_nodes), modifier=0.7)

        template.ax.scatter(interpolation_nodes, interpolation_values, color='red', s=size, label="Interpolation nodes",
                            zorder=10)



    @classmethod
    def _plot_segments_(cls, template: PlotTemplate, data: _Data_) -> None:
        for function_index, segments in enumerate(data.connectable_segments):
            line_width, line_style, color, alpha, z_order = cls._create_plot_parameters_(function_index, data)

            for plot_segment in segments[0]:
                x_values = jnp.array([data.evaluation_points[index] for index in plot_segment])
                y_values = jnp.array([data.function_values[function_index][index] for index in plot_segment])

                template.ax.plot(x_values, y_values, linewidth=line_width, linestyle=line_style, color=color,
                                 alpha=alpha, zorder=z_order)

            for scatter_segment in segments[1]:
                left_segment_border = data.evaluation_points[scatter_segment[0]]

                scatter_x_values = [left_segment_border]

                if len(scatter_segment) > 1:
                    right_segment_border = data.evaluation_points[scatter_segment[-1]]
                    scatter_x_values.append(right_segment_border)

                    amount_of_intermediate_points = jnp.floor((right_segment_border - left_segment_border) / data.scatter_x_distance)
                    intermediate_points = left_segment_border + jnp.arange(1, amount_of_intermediate_points + 1) * data.scatter_x_distance
                    scatter_x_values.extend(intermediate_points)

                scatter_x_values = jnp.array(scatter_x_values)
                # compiled_function = data.functions[function_index].compile(len(scatter_x_values), left_segment_border.dtype)
                # actual_y_values: jnp.ndarray = compiled_function.evaluate(scatter_x_values)
                actual_y_values = PlotUtils.evaluate_function(data.functions[function_index], data.data_types[function_index], scatter_x_values)
                actual_y_values = PlotUtils.clamp_function_values(actual_y_values, data.y_limits)

                for x_value, actual_y_value in zip(scatter_x_values, actual_y_values):

                    if jnp.isneginf(actual_y_value):
                        y_value = data.y_limits[0] - function_index * data.scatter_y_distance
                        template.ax.scatter(x_value, y_value, color=color, s=data.scatter_size, zorder=10)
                    else:
                        y_value = data.y_limits[1] + function_index * data.scatter_y_distance
                        template.ax.scatter(x_value, y_value, color=color, s=data.scatter_size, zorder=10)



    @classmethod
    def _create_plot_parameters_(cls, function_index: int, data: _Data_) -> tuple[float, str, str, float, int]:
        colors = InterpolantsPlotComponentConstants.COLORS
        line_styles = cls._create_line_styles_(data)

        line_width: float = 4 if function_index == 0 else 2
        line_style: str = line_styles[function_index]
        color: str = colors[function_index % len(colors)]
        alpha: float = 1 #0.6 if function_index == 0 else 1
        z_order: int = function_index

        return line_width, line_style, color, alpha, z_order



    @staticmethod
    def _create_line_styles_(data: _Data_) -> list:
        d = InterpolantsPlotComponentConstants.LINE_STYLE_DASH_DISTANCE
        line_styles = ["-"]
        for i in range(data.amount_of_functions_to_plot - 1):
            style = (i * d, (d, (data.amount_of_functions_to_plot - 2) * d))
            line_styles.append(style)

        return line_styles



    @classmethod
    def _set_legend_(cls, template: PlotTemplate, data: _Data_) -> None:
        custom_lines: list = [Line2D([0], [0], color='red', marker='o', markersize=8, linestyle='None')]
        labels = ["Nodes"]

        for function_index in range(data.amount_of_functions_to_plot):
            line_width, line_style, color, alpha, z_order = cls._create_plot_parameters_(function_index, data)
            label: str = data.function_names[function_index]

            line = Line2D([0], [0], color=color, linewidth=line_width)
            custom_lines.append(line)
            labels.append(label)

        template.ax.legend(custom_lines, labels)








