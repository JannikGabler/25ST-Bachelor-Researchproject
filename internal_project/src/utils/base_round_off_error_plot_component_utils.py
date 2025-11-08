import math

import jax.numpy as jnp

from fractions import Fraction

from matplotlib.lines import Line2D

from constants.internal_logic_constants import BaseRoundOffErrorPlotComponentConstants
from data_classes.pipeline_data.pipeline_data import PipelineData
from data_classes.plot_template.plot_template import PlotTemplate
from data_classes.plotting.base_round_off_error_plot_component_utils_data.base_round_off_error_plot_component_utils_data import BaseRoundOffErrorPlotComponentUtilsData
from exceptions.not_instantiable_error import NotInstantiableError
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from utils.plot_utils import PlotUtils


class BaseRoundOffErrorPlotComponentUtils:
    """
    Base utility helpers for creating round-off error plots. This class cannot be instantiated.
    """


    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        """
        Raises:
            NotInstantiableError: Always raised when initialized to indicate that this class is not meant to be instantiated.
        """

        raise NotInstantiableError(f"The class {repr(self.__class__.__name__)} can not be instantiated.")


    ######################
    ### Public methods ###
    ######################
    @classmethod
    def plot_data(cls, pipeline_data: list[PipelineData], additional_data: AdditionalComponentExecutionData, set_error_callable: callable) -> PlotTemplate:
        """
        Create the base round-off error plot for the given pipeline data.

        Args:
            pipeline_data (list[PipelineData]): Data containing interpolant functions to evaluate.
            additional_data (AdditionalComponentExecutionData): Additional information for plot annotation.
            set_error_callable (callable): Function used to compute specific round-off error values.

        Returns:
            PlotTemplate: Configured plot visualizing the computed round-off errors.
        """

        data: BaseRoundOffErrorPlotComponentUtilsData = (BaseRoundOffErrorPlotComponentUtilsData())
        template: PlotTemplate = PlotTemplate(figsize=BaseRoundOffErrorPlotComponentConstants.FIGURE_SIZE)
        data.amount_of_functions_to_plot = len(pipeline_data)
        cls._set_evaluation_points_(data, pipeline_data)
        cls._set_exact_interpolation_nodes_(data, pipeline_data)
        cls._set_exact_interpolation_values_(data, pipeline_data)
        cls._set_barycentric_weights_(data, pipeline_data)
        cls._set_exact_interpolant_values_(data, pipeline_data)
        cls._set_functions_(data, pipeline_data)
        cls._set_function_names_(data, pipeline_data)
        set_error_callable(data, pipeline_data)
        cls._set_y_threshold_(data, additional_data)
        cls._set_y_limit_(data, additional_data)
        cls._clean_up_errors_(data, pipeline_data)
        cls._set_scatter_size_and_distances_(template, data, pipeline_data)
        cls._set_border_(data, pipeline_data)
        cls._set_connectable_segments_(data)
        cls._plot_segments_(template, data)
        cls._draw_y_threshold_line_(template, data)
        template.ax.scatter(data.evaluation_points[0], data.border[1][0], alpha=0)
        template.ax.scatter(data.evaluation_points[0], data.border[1][1], alpha=0)
        meta_info_str: str = PlotUtils.create_plot_meta_info_str(pipeline_data, additional_data)
        template.ax.set_title(meta_info_str, fontsize=10)
        cls._set_legend_(template, data)
        template.ax.grid()
        return template


    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _set_evaluation_points_(data: BaseRoundOffErrorPlotComponentUtilsData, pipeline_data: list[PipelineData]) -> None:
        pd: PipelineData = pipeline_data[0]
        data.evaluation_points = PlotUtils.create_plot_points(pd.interpolation_interval, BaseRoundOffErrorPlotComponentConstants.AMOUNT_OF_EVALUATION_POINTS)
        data.evaluation_points_exact = PlotUtils.create_exact_plot_points(pd.interpolation_interval, BaseRoundOffErrorPlotComponentConstants.AMOUNT_OF_EVALUATION_POINTS)


    @staticmethod
    def _set_exact_interpolation_nodes_(data: BaseRoundOffErrorPlotComponentUtilsData, pipeline_data: list[PipelineData]) -> None:
        pd: PipelineData = pipeline_data[0]
        data.interpolation_nodes_exact = [Fraction.from_float(node.astype(float).item()) for node in pd.interpolation_nodes]


    @staticmethod
    def _set_exact_interpolation_values_(data: BaseRoundOffErrorPlotComponentUtilsData, pipeline_data: list[PipelineData]) -> None:
        pd: PipelineData = pipeline_data[0]
        data.interpolation_values_exact = [Fraction.from_float(value.astype(float).item()) for value in pd.interpolation_values]


    @staticmethod
    def _set_barycentric_weights_(data: BaseRoundOffErrorPlotComponentUtilsData, pipeline_data: list[PipelineData]) -> None:
        pd: PipelineData = pipeline_data[0]

        data.barycentric_weights_exact = []
        for i in range(pd.node_count):
            denominator: Fraction = Fraction(1)

            for j in range(pd.node_count):
                if i != j:
                    denominator *= (data.interpolation_nodes_exact[i] - data.interpolation_nodes_exact[j])

            data.barycentric_weights_exact.append(1 / denominator)


    @staticmethod
    def _set_exact_interpolant_values_(data: BaseRoundOffErrorPlotComponentUtilsData, pipeline_data: list[PipelineData]) -> None:
        pd: PipelineData = pipeline_data[0]

        data.interpolant_values_exact = []

        for exact_evaluation_point in data.evaluation_points_exact:
            match_index: int | None = next((index for index, value in enumerate(data.interpolation_nodes_exact) if value == exact_evaluation_point), None)

            if match_index is not None:
                data.interpolant_values_exact.append(data.interpolation_values_exact[match_index])
            else:
                nominator: Fraction = Fraction(0)
                denominator: Fraction = Fraction(0)

                for j in range(pd.node_count):
                    value: Fraction = data.barycentric_weights_exact[j] / (exact_evaluation_point - data.interpolation_nodes_exact[j])

                    nominator += data.interpolation_values_exact[j] * value
                    denominator += value

                data.interpolant_values_exact.append(nominator / denominator)


    @staticmethod
    def _set_functions_(data: BaseRoundOffErrorPlotComponentUtilsData, pipeline_data: list[PipelineData]) -> None:
        data.functions = [pd.interpolant for pd in pipeline_data]


    @staticmethod
    def _set_function_names_(data: BaseRoundOffErrorPlotComponentUtilsData, pipeline_data: list[PipelineData]) -> None:
        data.function_names = [pd.interpolant.name for pd in pipeline_data]


    @staticmethod
    def _set_y_threshold_(data: BaseRoundOffErrorPlotComponentUtilsData, additional_data: AdditionalComponentExecutionData) -> None:
        y_threshold: object = additional_data.overridden_attributes.get(BaseRoundOffErrorPlotComponentConstants.Y_THRESHOLD_ATTRIBUTE_NAME, BaseRoundOffErrorPlotComponentConstants.DEFAULT_Y_THRESHOLD)

        if not isinstance(y_threshold, float):
            raise TypeError(f"The attribute {repr(BaseRoundOffErrorPlotComponentConstants.Y_THRESHOLD_ATTRIBUTE_NAME)}"
                            f"for the component {additional_data.own_graph_node.value.component_name} must be a float.")

        data.y_threshold = y_threshold


    @staticmethod
    def _set_y_limit_(data: BaseRoundOffErrorPlotComponentUtilsData, additional_data: AdditionalComponentExecutionData) -> None:
        y_limit: object = additional_data.overridden_attributes.get(BaseRoundOffErrorPlotComponentConstants.Y_LIMIT_ATTRIBUTE_NAME, BaseRoundOffErrorPlotComponentConstants.DEFAULT_Y_LIMIT)

        if not isinstance(y_limit, float):
            raise TypeError(f"The attribute {repr(BaseRoundOffErrorPlotComponentConstants.Y_LIMIT_ATTRIBUTE_NAME)}"
                            f"for the component {additional_data.own_graph_node.value.component_name} must be a float.")

        data.y_limit = y_limit


    @staticmethod
    def _clean_up_errors_(data: BaseRoundOffErrorPlotComponentUtilsData, pipeline_data: list[PipelineData]) -> None:
        pd: PipelineData = pipeline_data[0]
        y_limits = (jnp.array(-jnp.inf, dtype=pd.data_type), jnp.array(data.y_limit, dtype=pd.data_type))

        for i, absolute_round_off_error in enumerate(data.round_off_errors):
            cleaned_values = PlotUtils.replace_nan_with_inf(absolute_round_off_error)
            cleaned_values = PlotUtils.clamp_function_values(cleaned_values, y_limits)
            data.round_off_errors[i] = cleaned_values


    @classmethod
    def _set_scatter_size_and_distances_(cls, template: PlotTemplate, data: BaseRoundOffErrorPlotComponentUtilsData, pipeline_data: list[PipelineData]) -> None:
        pd: PipelineData = pipeline_data[0]
        data.scatter_size = 9 * PlotUtils.scatter_size_for_equidistant_circles(BaseRoundOffErrorPlotComponentConstants.AMOUNT_OF_EVALUATION_POINTS, BaseRoundOffErrorPlotComponentConstants.FIGURE_SIZE[0])
        data.scatter_x_distance = PlotUtils.scatter_distance_for_equidistant_circles(BaseRoundOffErrorPlotComponentConstants.AMOUNT_OF_EVALUATION_POINTS, pd.interpolation_interval[1] - pd.interpolation_interval[0])
        data.scatter_y_distance = cls._calc_scatter_y_distance_(template, data)


    @staticmethod
    def _calc_scatter_y_distance_(template: PlotTemplate, data: BaseRoundOffErrorPlotComponentUtilsData) -> float:
        scatter_y_distance_inch: float = math.sqrt(data.scatter_size / math.pi) / 72
        height_in_inches: float = (BaseRoundOffErrorPlotComponentConstants.FIGURE_SIZE[1] / template.fig.dpi)

        numerator = data.y_limit
        denominator = (100 * height_in_inches / scatter_y_distance_inch - data.amount_of_functions_to_plot)
        return 3 * numerator / denominator


    @staticmethod
    def _set_border_(data: BaseRoundOffErrorPlotComponentUtilsData, pipeline_data: list[PipelineData]) -> None:
        pd: PipelineData = pipeline_data[0]
        upper_y_border = (data.y_limit + data.amount_of_functions_to_plot * data.scatter_y_distance)
        data.border = (pd.interpolation_interval, (0, upper_y_border))


    @classmethod
    def _set_connectable_segments_(cls, data: BaseRoundOffErrorPlotComponentUtilsData) -> None:
        data.connectable_segments = []
        for i, error in enumerate(data.round_off_errors):
            segments_to_plot, segments_to_scatter = (cls._extract_indices_of_connectable_segments_(error))
            data.connectable_segments.append((segments_to_plot, segments_to_scatter))


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


    @classmethod
    def _plot_segments_(cls, template: PlotTemplate, data: BaseRoundOffErrorPlotComponentUtilsData) -> None:
        for function_index, segments in enumerate(data.connectable_segments):
            y_value_for_inf: float = (data.y_limit + function_index * data.scatter_y_distance)
            line_width, line_style, color, alpha, z_order = (cls._create_plot_parameters_(function_index, data))

            for plot_segment in segments[0]:
                x_values = jnp.array([data.evaluation_points[index] for index in plot_segment])
                y_values = jnp.array([data.round_off_errors[function_index][index] for index in plot_segment])

                template.ax.plot(x_values, y_values, linewidth=line_width, linestyle=line_style, color=color, alpha=alpha, zorder=z_order)

            for scatter_segment in segments[1]:
                x_values = jnp.array([data.evaluation_points[index] for index in scatter_segment])
                y_values = jnp.array([y_value_for_inf for _ in scatter_segment])
                template.ax.scatter(x_values, y_values, s=data.scatter_size, color=color, zorder=data.amount_of_functions_to_plot)


    @classmethod
    def _create_plot_parameters_(cls, function_index: int, data: BaseRoundOffErrorPlotComponentUtilsData) -> tuple[float, str, str, float, int]:
        colors = BaseRoundOffErrorPlotComponentConstants.COLORS
        line_styles = cls._create_line_styles_(data)

        line_width: float = BaseRoundOffErrorPlotComponentConstants.LINE_WIDTH
        line_style: str = line_styles[function_index]
        color: str = colors[function_index % len(colors)]
        alpha: float = 1
        z_order: int = function_index

        return line_width, line_style, color, alpha, z_order


    @staticmethod
    def _create_line_styles_(data: BaseRoundOffErrorPlotComponentUtilsData) -> list:
        d = BaseRoundOffErrorPlotComponentConstants.LINE_STYLE_DASH_DISTANCE
        line_styles = ["-"]
        for i in range(data.amount_of_functions_to_plot - 1):
            style = (i * d, (d, (data.amount_of_functions_to_plot - 2) * d))
            line_styles.append(style)

        return line_styles


    @staticmethod
    def _draw_y_threshold_line_(template: PlotTemplate, data: BaseRoundOffErrorPlotComponentUtilsData) -> None:
        template.ax.plot(data.border[0], [data.y_threshold, data.y_threshold], color="red", linestyle="--", linewidth=1, zorder=data.amount_of_functions_to_plot + 1)


    @classmethod
    def _set_legend_(cls, template: PlotTemplate, data: BaseRoundOffErrorPlotComponentUtilsData) -> None:
        custom_lines: list = [Line2D([0], [0], color="red", linewidth=1, linestyle="--")]
        labels = ["Threshold"]

        for function_index in range(data.amount_of_functions_to_plot):
            line_width, line_style, color, alpha, z_order = (cls._create_plot_parameters_(function_index, data))
            label: str = data.function_names[function_index]

            line = Line2D([0], [0], color=color, linewidth=line_width)
            custom_lines.append(line)
            labels.append(label)

        template.ax.legend(custom_lines, labels)
