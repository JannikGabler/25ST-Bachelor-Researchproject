import jax.numpy as jnp
from jax import Array

from constants.internal_logic_constants import OldInterpolantsPlotComponentConstants
from functions.abstracts.compilable_function import CompilableFunction
from functions.abstracts.compiled_function import CompiledFunction
from data_classes.pipeline_data.pipeline_data import PipelineData
from data_classes.plotting_data.function_plot_data import FunctionPlotData
from data_classes.plot_template.plot_template import PlotTemplate
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import (
    AdditionalComponentExecutionData,
)
from utils.plot_utils import PlotUtils


class OldInterpolantsPlotComponentUtils:
    """
    Plots original function + interpolants using PlotTemplate (deferred rendering).
    """

    ######################
    ### Public methods ###
    ######################
    @classmethod
    def plot_data(
        cls,
        pipeline_data: list[PipelineData],
        additional_data: AdditionalComponentExecutionData,
    ) -> PlotTemplate:
        plot_points, functions, y_limits = cls._create_plot_data_(pipeline_data)

        template: PlotTemplate = cls._init_plot_()
        cls._plot_interpolation_points_(template, pipeline_data[0])

        for i, function in enumerate(functions):
            cls._plot_function_(template, plot_points, function, i, y_limits)

        cls._finish_up_plot_(template, pipeline_data, additional_data)
        return template

    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _init_plot_() -> PlotTemplate:
        # keep previous (10, 6) figure size
        return PlotTemplate(figsize=(10, 6))

    @classmethod
    def _plot_interpolation_points_(
        cls, template: PlotTemplate, pipeline_data: PipelineData
    ) -> None:
        interpolation_nodes: jnp.ndarray = pipeline_data.interpolation_nodes
        interpolation_values: jnp.ndarray = pipeline_data.interpolation_values

        size = PlotUtils.adaptive_scatter_size(len(interpolation_nodes), modifier=0.7)

        template.ax.scatter(
            interpolation_nodes,
            interpolation_values,
            color="red",
            s=size,
            label="Interpolation nodes",
            zorder=10,
        )

    @classmethod
    def _create_plot_data_(
        cls, pipeline_data: list[PipelineData]
    ) -> tuple[Array, list[CompilableFunction], tuple[Array, Array]]:
        main_data: PipelineData = pipeline_data[0]
        interval: jnp.ndarray = main_data.interpolation_interval

        plot_points: jnp.ndarray = PlotUtils.create_plot_points(
            interval, OldInterpolantsPlotComponentConstants.AMOUNT_OF_EVALUATION_POINTS
        )

        functions: list[CompilableFunction] = [main_data.original_function] + [
            data.interpolant for data in pipeline_data
        ]

        y_limits: tuple[jnp.ndarray, jnp.ndarray] = cls._calc_y_limits_(
            functions[0],
            plot_points,
            OldInterpolantsPlotComponentConstants.Y_LIMIT_FACTOR,
        )

        return plot_points, functions, y_limits

    @classmethod
    def _calc_y_limits_(
        cls, function: CompilableFunction, plot_points: jnp.ndarray, y_factor: float
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        compiled_function: CompiledFunction = function.compile(
            len(plot_points), plot_points.dtype
        )
        function_values: jnp.ndarray = compiled_function.evaluate(plot_points)

        valid_mask: jnp.ndarray = jnp.isfinite(function_values)
        clean_values = function_values[valid_mask]
        min_value = jnp.min(clean_values)
        max_value = jnp.max(clean_values)
        difference = y_factor * (max_value - min_value)
        return min_value - difference, max_value + difference

    @classmethod
    def _plot_function_(
        cls,
        template: PlotTemplate,
        plot_points: jnp.ndarray,
        function: CompilableFunction,
        function_index: int,
        y_limits: tuple[jnp.ndarray, jnp.ndarray],
    ) -> None:

        function_plot_data: FunctionPlotData = (
            PlotUtils.create_plot_data_from_compilable_function(
                function, function_index, plot_points, y_limits
            )
        )

        connectable_segments = function_plot_data.connectable_segments
        single_points = function_plot_data.single_points

        line_width, line_style, color, alpha, z_order, label = (
            cls._create_plot_parameters_(function_plot_data, function_index)
        )
        label_used: bool = False

        for segment in connectable_segments:
            x_array, y_array = zip(*segment)
            template.ax.plot(
                x_array,
                y_array,
                linewidth=line_width,
                linestyle=line_style,
                color=color,
                alpha=alpha,
                label=label if not label_used else None,
                zorder=z_order,
            )
            label_used = True

        for point in single_points:
            template.ax.scatter(
                point[0],
                point[1],
                color=color,
                s=PlotUtils.adaptive_scatter_size(len(plot_points)),
                label=label if not label_used else None,
                zorder=10,
            )
            label_used = True

    @staticmethod
    def _create_plot_parameters_(
        function_plot_data: FunctionPlotData, function_index: int
    ) -> tuple[float, str, str, float, int, str]:
        colors = OldInterpolantsPlotComponentConstants.COLORS
        line_styles = OldInterpolantsPlotComponentConstants.LINE_STYLES

        line_width: float = 4 if function_index == 0 else 2
        line_style: str = (
            "-"
            if function_index == 0
            else line_styles[(function_index - 1) % len(line_styles)]
        )
        color: str = colors[function_index % len(colors)]
        alpha: float = 0.6 if function_index == 0 else 1
        z_order: int = 1 if function_index == 0 else 2
        label: str = function_plot_data.function_name

        return line_width, line_style, color, alpha, z_order, label

    @staticmethod
    def _finish_up_plot_(
        template: PlotTemplate,
        pipeline_data: list[PipelineData],
        additional_data: AdditionalComponentExecutionData,
    ) -> None:
        meta_info_str: str = PlotUtils.create_plot_meta_info_str(
            pipeline_data, additional_data
        )

        template.fig.suptitle("Interpolant plot")
        template.ax.set_title(meta_info_str, fontsize=10)
        template.ax.set_xlabel("x")
        template.ax.set_ylabel("f(x)")

        template.ax.legend()
        template.ax.grid(True)
        template.fig.tight_layout()
