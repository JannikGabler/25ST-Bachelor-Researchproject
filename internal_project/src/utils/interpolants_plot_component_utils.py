import jax.numpy as jnp

from matplotlib import pyplot as plt

from constants.internal_logic_constants import InterpolantsPlotComponentConstants
from pipeline_entities.large_data_classes.plotting_data.function_plot_data import FunctionPlotData
from pipeline_entities.large_data_classes.plotting_data.interpolants_plot_component_data import \
    InterpolantsPlotComponentData
from utils.plot_utils import PlotUtils


class InterpolantsPlotComponentUtils:




        # #InterpolantsPlotComponentUtils.init_plot()
        #
        # InterpolantsPlotComponentUtils.plot_interpolation_points(
        #     pipeline_data[0].interpolation_nodes, pipeline_data[0].interpolation_values
        # )
        # InterpolantsPlotComponentUtils.plot_original_function(x_array, original_function_values, y_limits)
        # InterpolantsPlotComponentUtils.plot_interpolants(pipeline_data, x_array, y_limits)
        #
        # InterpolantsPlotComponentUtils.finish_up_plot(pipeline_data, additional_execution_info)
        #
        # meta_info_str: str = PlotUtils.create_plot_meta_info_str(pipeline_data, additional_execution_info)
        # plot_data: InterpolantsPlotData = InterpolantsPlotData(meta_info_str, plot_points, )



    @classmethod
    def plot_data(cls, data: InterpolantsPlotComponentData) -> None:
        amount_of_plot_points: int = len(data.plot_points)

        cls._init_plot_()
        cls._plot_interpolation_points_(data)

        plot_data_list: list[FunctionPlotData] = [data.original_function_plot_data] + data.interpolants_plot_data

        for plot_data in plot_data_list:
            cls._plot_function_(plot_data, amount_of_plot_points)


        cls._finish_up_plot_(data)
        plt.show(block=True)



    @staticmethod
    def _init_plot_():
        plt.figure(figsize=(10, 6))



    @classmethod
    def _plot_interpolation_points_(cls, data: InterpolantsPlotComponentData) -> None:
        interpolation_nodes: jnp.ndarray = data.interpolation_nodes
        interpolation_values: jnp.ndarray = data.interpolation_values

        size = PlotUtils.adaptive_scatter_size(len(interpolation_nodes), modifier=0.7)

        plt.scatter(interpolation_nodes, interpolation_values, color='red', s=size, label="Interpolation Nodes",
                    zorder=10)



    @classmethod
    def _plot_function_(cls, function_plot_data: FunctionPlotData, amount_of_plot_points: int) -> None:
        index: int = function_plot_data.function_index
        connectable_segments = function_plot_data.connectable_segments
        single_points = function_plot_data.single_points
        colors = InterpolantsPlotComponentConstants.COLORS
        line_styles = InterpolantsPlotComponentConstants.LINE_STYLES

        line_width: float = 4 if index == 0 else 2
        line_style: str = '-' if index == 0 else line_styles[(index - 1) % len(line_styles)]
        color: str = colors[index % len(colors)]
        alpha: float = 0.6 if index == 0 else 1
        z_order: int = 1 if index == 0 else 2
        label: str = function_plot_data.function_name
        label_used: bool = False

        for segment in connectable_segments:
            x_array, y_array = zip(*segment)
            plt.plot(x_array, y_array, linewidth=line_width, linestyle=line_style, color=color, alpha=alpha,
                     label=label if not label_used else None, zorder=z_order)
            label_used = True

        for point in single_points:
            plt.scatter(point[0], point[1], color=color, s=PlotUtils.adaptive_scatter_size(amount_of_plot_points),
                        label=label if not label_used else None, zorder=10)
            label_used = True



    # @classmethod
    # def _plot_original_function_(cls, x_array: jnp.ndarray, function_values: jnp.ndarray, y_limits: tuple[jnp.ndarray, jnp.ndarray]) -> None:
    #
    #     x_intervals, y_intervals, inf_indices = cls._split_up_function_values_(x_array, function_values)
    #
    #     for x_interval, y_interval in zip(x_intervals, y_intervals):
    #         plt.plot(x_interval, y_interval, label="Original Function", linewidth=4, color=cls.COLORS[0], alpha=0.6,
    #                  zorder=1)
    #
    #     for inf_index in inf_indices:
    #         value: jnp.ndarray = function_values[inf_index]
    #         clamped_value = y_limits[0] if jnp.isneginf(value) else y_limits[1]
    #         size = PlotUtils.adaptive_scatter_size(len(x_array))
    #         plt.scatter(x_array[inf_index], clamped_value, color=cls.COLORS[0], s=size, zorder=10)



    # @classmethod
    # def plot_interpolants(cls, pipeline_data: list[PipelineData], x_array: jnp.ndarray,
    #                       y_limits: tuple[jnp.ndarray, jnp.ndarray]) -> None:
    #
    #     for i, data in enumerate(pipeline_data):
    #         if data.interpolant is not None:
    #             interpolant: CompiledInterpolant = data.interpolant.compile(len(x_array), data.data_type)
    #             function_values: jnp.ndarray = interpolant.evaluate(x_array.astype(data.data_type))
    #             function_values = PlotUtils.replace_nan_with_inf(function_values)
    #             function_values = PlotUtils.clamp_function_values(function_values, y_limits)
    #
    #             x_intervals, y_intervals, inf_indices = cls._split_up_function_values_(x_array, function_values)
    #
    #             #label: str = f"{data.interpolant.name} (d={data.data_type.__name__}, n={data.node_count}, i={data.interpolation_interval})"
    #
    #             plt.plot([], [], label=data.interpolant.name, color=cls.COLORS[i+1])
    #
    #             for j, (x_interval, y_interval) in enumerate(zip(x_intervals, y_intervals)):
    #                 plt.plot(x_interval, y_interval, linewidth=2, color=cls.COLORS[i+1 % len(cls.COLORS)],
    #                          linestyle=cls.LINE_STYLES[i % len(cls.LINE_STYLES)], zorder=2)
    #
    #             for inf_index in inf_indices:
    #                 value: jnp.ndarray = function_values[inf_index]
    #                 clamped_value = y_limits[0] - i/4 if jnp.isneginf(value) else y_limits[1] + i/4
    #                 size = PlotUtils.adaptive_scatter_size(len(x_array))
    #                 plt.scatter(x_array[inf_index], clamped_value, color=cls.COLORS[i+1], s=size, zorder=10)














    @staticmethod
    def _finish_up_plot_(data: InterpolantsPlotComponentData) -> None:
        plt.suptitle(f"Interpolant plot")
        plt.title(data.meta_info_str, fontsize=10)
        plt.xlabel("x")
        plt.ylabel("f(x)")

        plt.legend()

        # plt.legend(
        #     loc="upper center",
        #     bbox_to_anchor=(0.5, -0.1),
        #     ncol=2  # mehrere Spalten nebeneinander
        # )
        plt.grid(True)
        plt.tight_layout()

