import math

import jax.numpy as jnp

from dataclasses import dataclass
from fractions import Fraction

from matplotlib.lines import Line2D

from constants.internal_logic_constants import AbsoluteRoundOffErrorPlotComponentConstants
from data_classes.pipeline_data.pipeline_data import PipelineData
from data_classes.plot_template.plot_template import PlotTemplate
from exceptions.not_instantiable_error import NotInstantiableError
from functions.abstracts.compilable_function import CompilableFunction
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import \
    AdditionalComponentExecutionData
from utils.plot_utils import PlotUtils


class AbsoluteRoundOffErrorPlotComponentUtils:

    @dataclass
    class _Data_:
        amount_of_functions_to_plot: int | None = None

        evaluation_points: jnp.ndarray | None = None
        evaluation_points_exact: list[Fraction] | None = None

        interpolation_nodes_exact: list[Fraction] | None = None
        interpolation_values_exact: list[Fraction] | None = None

        barycentric_weights_exact: list[Fraction] | None = None
        interpolant_values_exact: list[Fraction] | None = None

        functions: list[CompilableFunction] | None = None
        function_names: list[str] | None = None

        absolute_round_off_errors: list[jnp.ndarray] | None = None

        y_threshold: float | None = None
        y_limit: float | None = None
        scatter_size: float | None = None
        scatter_x_distance: jnp.ndarray | None = None
        scatter_y_distance: jnp.ndarray | None = None
        border: tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]] | None = None
        connectable_segments: list[tuple[list[list[int]], list[list[int]]]] | None = None



    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        raise NotInstantiableError(f"The class {repr(self.__class__.__name__)} can not be instantiated.")



    ######################
    ### Public methods ###
    ######################
    @classmethod
    def plot_data(cls, pipeline_data: list[PipelineData], additional_data: AdditionalComponentExecutionData) -> PlotTemplate:
        pd: PipelineData = pipeline_data[0]
        data: cls._Data_ = cls._Data_()
        template: PlotTemplate = PlotTemplate(figsize=AbsoluteRoundOffErrorPlotComponentConstants.FIGURE_SIZE)

        data.amount_of_functions_to_plot = len(pipeline_data) + 1

        cls._set_evaluation_points_(data, pipeline_data)

        cls._set_exact_interpolation_nodes_(data, pipeline_data)
        cls._set_exact_interpolation_values_(data, pipeline_data)

        cls._set_barycentric_weights_(data, pipeline_data)

        cls._set_functions_(data, pipeline_data)
        cls._set_function_names_(data, pipeline_data)

        cls._set_absolute_round_off_errors_(data, pipeline_data)

        cls._set_y_threshold_(data, additional_data)
        cls._set_y_limit_(data, additional_data)

        cls._clean_up_errors_(data, pipeline_data)

        cls._set_scatter_size_and_distances_(template, data, pipeline_data)

        cls._set_border_(data, pipeline_data)

        cls._set_connectable_segments_(data)

        cls._plot_segments_(template, data)

        template.ax.scatter(data.evaluation_points[0], data.border[1][0], alpha=0)
        template.ax.scatter(data.evaluation_points[0], data.border[1][1], alpha=0)

        meta_info_str: str = PlotUtils.create_plot_meta_info_str(pipeline_data, additional_data)

        template.fig.suptitle(f"Absolute round-off error plot")
        template.ax.set_title(meta_info_str, fontsize=10)
        template.ax.set_xlabel("x")
        template.ax.set_ylabel("y")

        cls._set_legend_(template, data)
        template.ax.grid()
        template.fig.tight_layout()

        return template



    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _set_evaluation_points_(data: _Data_, pipeline_data: list[PipelineData]) -> None:
        pd: PipelineData = pipeline_data[0]

        data.evaluation_points = PlotUtils.create_plot_points(pd.interpolation_interval,
             AbsoluteRoundOffErrorPlotComponentConstants.AMOUNT_OF_EVALUATION_POINTS, pd.data_type)

        data.evaluation_points_exact = PlotUtils.create_exact_plot_points(pd.interpolation_interval,
            AbsoluteRoundOffErrorPlotComponentConstants.AMOUNT_OF_EVALUATION_POINTS)



    @staticmethod
    def _set_exact_interpolation_nodes_(data: _Data_, pipeline_data: list[PipelineData]) -> None:
        pd: PipelineData = pipeline_data[0]

        data.interpolation_nodes_exact = [Fraction.from_float(node.astype(float).item()) for node in pd.interpolation_nodes]



    @staticmethod
    def _set_exact_interpolation_values_(data: _Data_, pipeline_data: list[PipelineData]) -> None:
        pd: PipelineData = pipeline_data[0]

        data.interpolation_values_exact = [Fraction.from_float(value.astype(float).item()) for value in pd.interpolation_values]



    @staticmethod
    def _set_barycentric_weights_(data: _Data_, pipeline_data: list[PipelineData]) -> None:
        pd: PipelineData = pipeline_data[0]

        data.barycentric_weights_exact = []
        for i in range(pd.node_count):
            denominator: Fraction = Fraction(1)

            for j in range(pd.node_count):
                if i != j:
                    denominator *= data.interpolation_nodes_exact[i] - data.interpolation_nodes_exact[j]

            data.barycentric_weights_exact.append(denominator)



    @staticmethod
    def _set_exact_interpolant_values_(data: _Data_, pipeline_data: list[PipelineData]) -> None:
        pd: PipelineData = pipeline_data[0]

        data.interpolant_values_exact = []

        for exact_evaluation_point in data.evaluation_points_exact:
            nominator: Fraction = Fraction(0)
            denominator: Fraction = Fraction(0)

            for i in range(pd.node_count):
                value: Fraction = data.barycentric_weights_exact[i] / (exact_evaluation_point - data.interpolation_nodes_exact[i])

                nominator += data.interpolation_values_exact[i] * value
                denominator += value

            data.interpolant_values_exact.append(nominator / denominator)



    @staticmethod
    def _set_functions_(data: _Data_, pipeline_data: list[PipelineData]) -> None:
        data.functions = [pd.interpolant for pd in pipeline_data]



    @staticmethod
    def _set_function_names_(data: _Data_, pipeline_data: list[PipelineData]) -> None:
        data.function_names = [pd.interpolant.name for pd in pipeline_data]



    @staticmethod
    def _set_absolute_round_off_errors_(data: _Data_, pipeline_data: list[PipelineData]) -> None:
        data.absolute_round_off_errors = []

        for i, pd in enumerate(pipeline_data):
            function_values: jnp.ndarray = PlotUtils.evaluate_function(pd.interpolant, data.evaluation_points)
            interpolant_values_float: list[float] = [float(value) for value in data.interpolant_values_exact]
            abs_round_off_errors: jnp.ndarray = jnp.abs(function_values - jnp.array(interpolant_values_float))
            data.absolute_round_off_errors.append(abs_round_off_errors)



    @staticmethod
    def _set_y_threshold_(data: _Data_, additional_data: AdditionalComponentExecutionData) -> None:
        y_threshold: object = additional_data.overridden_attributes.get(
            AbsoluteRoundOffErrorPlotComponentConstants.Y_THRESHOLD_ATTRIBUTE_NAME,
            AbsoluteRoundOffErrorPlotComponentConstants.DEFAULT_Y_THRESHOLD)

        if not isinstance(y_threshold, float):
            raise TypeError(f"The attribute {repr(AbsoluteRoundOffErrorPlotComponentConstants.Y_THRESHOLD_ATTRIBUTE_NAME)}"
                            f"for the component {additional_data.own_graph_node.value.component_name} must be a float.")

        data.y_threshold = y_threshold



    @staticmethod
    def _set_y_limit_(data: _Data_, additional_data: AdditionalComponentExecutionData) -> None:
        y_limit: object = additional_data.overridden_attributes.get(
            AbsoluteRoundOffErrorPlotComponentConstants.Y_LIMIT_ATTRIBUTE_NAME,
            AbsoluteRoundOffErrorPlotComponentConstants.DEFAULT_Y_LIMIT)

        if not isinstance(y_limit, float):
            raise TypeError(
                f"The attribute {repr(AbsoluteRoundOffErrorPlotComponentConstants.Y_LIMIT_ATTRIBUTE_NAME)}"
                f"for the component {additional_data.own_graph_node.value.component_name} must be a float.")

        data.y_limit = y_limit



    @staticmethod
    def _clean_up_errors_(data: _Data_, pipeline_data: list[PipelineData]) -> None:
        pd: PipelineData = pipeline_data[0]
        y_limits = (jnp.array(-jnp.inf, dtype=pd.data_type), jnp.array(data.y_limit, dtype=pd.data_type))

        for i, absolute_round_off_error in enumerate(data.absolute_round_off_errors):
            cleaned_values = PlotUtils.replace_nan_with_inf(absolute_round_off_error)
            cleaned_values = PlotUtils.clamp_function_values(cleaned_values, y_limits)
            data.absolute_round_off_errors[i] = cleaned_values



    @classmethod
    def _set_scatter_size_and_distances_(cls, template: PlotTemplate, data: _Data_, pipeline_data: list[PipelineData]) -> None:
        pd: PipelineData = pipeline_data[0]

        data.scatter_size = PlotUtils.scatter_size_for_equidistant_circles(
            AbsoluteRoundOffErrorPlotComponentConstants.AMOUNT_OF_EVALUATION_POINTS,
            AbsoluteRoundOffErrorPlotComponentConstants.FIGURE_SIZE[0])

        data.scatter_x_distance = PlotUtils.scatter_distance_for_equidistant_circles(
            AbsoluteRoundOffErrorPlotComponentConstants.AMOUNT_OF_EVALUATION_POINTS,
            pd.interpolation_interval[1] - pd.interpolation_interval[0])

        data.scatter_y_distance = cls._calc_scatter_y_distance_(template, data)



    @staticmethod
    def _calc_scatter_y_distance_(template: PlotTemplate, data: _Data_) -> float:
        scatter_y_distance_inch: float = math.sqrt(data.scatter_size / math.pi) / 72
        height_in_inches: float = AbsoluteRoundOffErrorPlotComponentConstants.FIGURE_SIZE[1] / template.fig.dpi

        numerator = data.y_limit
        denominator = 100 * height_in_inches / scatter_y_distance_inch - data.amount_of_functions_to_plot
        return 3 * numerator / denominator



    @staticmethod
    def _set_border_(data: _Data_, pipeline_data: list[PipelineData]) -> None:
        pd: PipelineData = pipeline_data[0]
        upper_y_border = data.y_limit + data.amount_of_functions_to_plot * data.scatter_y_distance
        data.border = (pd.interpolation_interval, (0, upper_y_border))



    @classmethod
    def _set_connectable_segments_(cls, data: _Data_) -> None:
        data.connectable_segments = []
        for i, error in enumerate(data.absolute_round_off_errors):
            segments_to_plot, segments_to_scatter = cls._extract_indices_of_connectable_segments_(error)
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
    def _plot_segments_(cls, template: PlotTemplate, data: _Data_) -> None:
        for function_index, segments in enumerate(data.connectable_segments):
            y_value_for_inf: float = data.y_limit + function_index * data.scatter_y_distance
            line_width, line_style, color, alpha, z_order = cls._create_plot_parameters_(function_index, data)

            for plot_segment in segments[0]:
                x_values = jnp.array([data.evaluation_points[index] for index in plot_segment])
                y_values = jnp.array([data.absolute_round_off_errors[function_index][index] for index in plot_segment])

                template.ax.plot(x_values, y_values, linewidth=line_width, linestyle=line_style, color=color,
                                 alpha=alpha, zorder=z_order)

            for scatter_segment in segments[1]:
                x_values = jnp.array([data.evaluation_points[index] for index in scatter_segment])
                y_values = jnp.array([y_value_for_inf for index in scatter_segment])

                template.ax.plot(x_values, y_values, linewidth=line_width, linestyle=line_style, color=color,
                                 alpha=alpha, zorder=z_order)



    @classmethod
    def _create_plot_parameters_(cls, function_index: int, data: _Data_) -> tuple[float, str, str, float, int]:
        colors = AbsoluteRoundOffErrorPlotComponentConstants.COLORS
        line_styles = cls._create_line_styles_(data)

        line_width: float = 4 if function_index == 0 else 2
        line_style: str = line_styles[function_index]
        color: str = colors[function_index % len(colors)]
        alpha: float = 1 #0.6 if function_index == 0 else 1
        z_order: int = function_index

        return line_width, line_style, color, alpha, z_order



    @staticmethod
    def _create_line_styles_(data: _Data_) -> list:
        d = AbsoluteRoundOffErrorPlotComponentConstants.LINE_STYLE_DASH_DISTANCE
        line_styles = ["-"]
        for i in range(data.amount_of_functions_to_plot - 1):
            style = (i * d, (d, (data.amount_of_functions_to_plot - 2) * d))
            line_styles.append(style)

        return line_styles



    @classmethod
    def _set_legend_(cls, template: PlotTemplate, data: _Data_) -> None:
        custom_lines: list = []
        labels = []

        for function_index in range(data.amount_of_functions_to_plot):
            line_width, line_style, color, alpha, z_order = cls._create_plot_parameters_(function_index, data)
            label: str = data.function_names[function_index]

            line = Line2D([0], [0], color=color, linewidth=line_width)
            custom_lines.append(line)
            labels.append(label)

        template.ax.legend(custom_lines, labels)