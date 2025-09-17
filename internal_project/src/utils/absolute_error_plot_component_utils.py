import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from matplotlib import pyplot as plt

from constants.internal_logic_constants import AbsoluteErrorPlotComponentConstants
from functions.abstracts.compilable_function import CompilableFunction
from data_classes.pipeline_data.pipeline_data import PipelineData
from data_classes.plotting_data.function_plot_data import FunctionPlotData
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import \
    AdditionalComponentExecutionData
from utils.plot_utils import PlotUtils


class AbsoluteErrorPlotComponentUtils:

    ######################
    ### Public methods ###
    ######################
    @classmethod
    def plot_data(cls, pipeline_data: list[PipelineData], additional_data: AdditionalComponentExecutionData) -> None:
        plot_points, absolut_errors_list, y_limits = cls._create_plot_data_(pipeline_data, additional_data)

        cls._init_plot_()
        # cls._plot_interpolation_points_(pipeline_data[0])

        for i, absolut_errors in enumerate(absolut_errors_list):
            cls._plot_absolute_error_values_(
                plot_points, absolut_errors, pipeline_data[i].interpolant.name, i, y_limits
            )

        cls._finish_up_plot_(pipeline_data, additional_data)
        plt.show(block=AbsoluteErrorPlotComponentConstants.SHOW_PLOT_IN_SEPARATE_PROCESS)



    #######################
    ### Private methods ###
    #######################
    @classmethod
    def _create_plot_data_(cls, pipeline_data: list[PipelineData], additional_data: AdditionalComponentExecutionData) \
            -> tuple[Array, list[Array], tuple[Array, Array]]:

        main_data: PipelineData = pipeline_data[0]
        interval: jnp.ndarray = main_data.interpolation_interval

        plot_points: jnp.ndarray = PlotUtils.create_plot_points(
            interval, AbsoluteErrorPlotComponentConstants.AMOUNT_OF_EVALUATION_POINTS)

        original_function_values: jnp.ndarray = PlotUtils.evaluate_function(main_data.original_function, jnp.float32, plot_points)

        absolut_errors_list: list[jnp.ndarray] = [ cls._calc_absolute_errors_(plot_points, original_function_values,
            data.interpolant, data.data_type) for data in pipeline_data ]

        y_limits: tuple[jnp.ndarray, jnp.ndarray] = cls._calc_y_limits_(absolut_errors_list, additional_data)

        return plot_points, absolut_errors_list, y_limits



    @classmethod
    def _calc_y_limits_(cls, absolute_errors_list: list[jnp.ndarray], additional_data: AdditionalComponentExecutionData) -> tuple[jnp.ndarray, jnp.ndarray]:
        specified_y_limit: object = additional_data.overridden_attributes.get("y_limit")

        if specified_y_limit is not None:
            return jnp.array(0), jnp.array(specified_y_limit)
        else:
            max_absolute_error: jnp.ndarray = jnp.max(jnp.concatenate(absolute_errors_list))
            return jnp.array(0), max_absolute_error



    @staticmethod
    def _init_plot_():
        plt.figure(figsize=(10, 6))



    @staticmethod
    def _calc_absolute_errors_(plot_points: jnp.ndarray, original_function_values: jnp.ndarray,
                               function: CompilableFunction, data_type: DTypeLike) -> jnp.ndarray:

        function_values: jnp.ndarray = PlotUtils.evaluate_function(function, data_type, plot_points)
        cast_function_values: jnp.ndarray = function_values.astype(jnp.float32)

        return jnp.absolute(original_function_values - cast_function_values)



    # @classmethod
    # def _plot_interpolation_points_(cls, pipeline_data: PipelineData) -> None:
    #     interpolation_nodes: jnp.ndarray = pipeline_data.interpolation_nodes
    #     interpolation_values: jnp.ndarray = pipeline_data.interpolation_values
    #
    #     size = PlotUtils.adaptive_scatter_size(len(interpolation_nodes), modifier=0.7)
    #
    #     plt.scatter(interpolation_nodes, interpolation_values, color='red', s=size, label="Interpolation nodes",
    #                 zorder=10)



    # @classmethod
    # def create_from(cls, pipeline_data: list[PipelineData], additional_execution_info: AdditionalComponentExecutionData)\
    #                 -> InterpolantsPlotComponentData:







    @classmethod
    def _plot_absolute_error_values_(cls, plot_points: jnp.ndarray, absolute_errors: jnp.ndarray,
                    function_name: str, function_index: int, y_limits: tuple[jnp.ndarray, jnp.ndarray]) -> None:

        PlotUtils.replace_nan_with_inf(absolute_errors)
        PlotUtils.clamp_function_values(absolute_errors, y_limits)

        function_plot_data: FunctionPlotData = PlotUtils.create_plot_data_from_function_values(
            function_name, function_index, plot_points, absolute_errors, y_limits
        )

        connectable_segments = function_plot_data.connectable_segments
        single_points = function_plot_data.single_points

        line_style, color, label = cls._create_plot_parameters_(function_plot_data, function_index)
        label_used: bool = False

        for segment in connectable_segments:
            x_array, y_array = zip(*segment)
            plt.plot(x_array, y_array, linewidth=AbsoluteErrorPlotComponentConstants.LINE_WIDTH,
                     linestyle=line_style, color=color, label=label if not label_used else None)
            label_used = True

        for point in single_points:
            plt.scatter(point[0], point[1], color=color, s=PlotUtils.adaptive_scatter_size(len(plot_points)),
                        label=label if not label_used else None, zorder=10)
            label_used = True



    @staticmethod
    def _create_plot_parameters_(function_plot_data: FunctionPlotData, function_index: int) -> tuple[str, str, str]:
        colors = AbsoluteErrorPlotComponentConstants.COLORS
        line_styles = AbsoluteErrorPlotComponentConstants.LINE_STYLES

        line_style: str = line_styles[function_index % len(line_styles)]
        color: str = colors[function_index % len(colors)]
        label: str = function_plot_data.function_name

        return line_style, color, label



    @staticmethod
    def _finish_up_plot_(pipeline_data: list[PipelineData], additional_data: AdditionalComponentExecutionData) -> None:
        meta_info_str: str = PlotUtils.create_plot_meta_info_str(pipeline_data, additional_data)

        plt.suptitle(f"Absolute error of interpolants plot")
        plt.title(meta_info_str, fontsize=10)
        plt.xlabel("x")
        plt.ylabel("$\\Delta$f(x)")

        overridden_y_scale: object = additional_data.overridden_attributes.get("y_scale")

        if overridden_y_scale is not None:
            overridden_y_scale_base: object = additional_data.overridden_attributes.get("y_scale_base")

            if overridden_y_scale_base is not None:
                plt.yscale(overridden_y_scale, base=overridden_y_scale_base)
            else:
                plt.yscale(overridden_y_scale)

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
