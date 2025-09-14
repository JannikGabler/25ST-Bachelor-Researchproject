import math
from fractions import Fraction

import jax.numpy as jnp
from numpy.typing import DTypeLike

from exceptions.not_instantiable_error import NotInstantiableError
from functions.abstracts.compilable_function import CompilableFunction
from functions.abstracts.compiled_function import CompiledFunction
from data_classes.pipeline_data.pipeline_data import PipelineData
from data_classes.plotting_data.function_plot_data import FunctionPlotData
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import \
    AdditionalComponentExecutionData


class PlotUtils:

    ###################
    ### Constructor ###
    ###################
    def __init__(self) -> None:
        raise NotInstantiableError(f"The class {repr(self.__class__.__name__)} cannot be instantiated.")




    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def create_plot_points(interval: jnp.ndarray, amount_of_evaluation_points: int, data_type: DTypeLike) -> jnp.ndarray:
        return jnp.linspace(interval[0], interval[1], amount_of_evaluation_points, dtype=data_type)



    @staticmethod
    def create_exact_plot_points(interval: jnp.ndarray, amount_of_evaluation_points: int) -> list[Fraction]:
        left_end: Fraction = Fraction.from_float(interval[0].astype(float).item())
        right_end: Fraction = Fraction.from_float(interval[1].astype(float).item())

        step_size: Fraction = (right_end - left_end) / (amount_of_evaluation_points - 1)
        return [left_end + i * step_size for i in range(amount_of_evaluation_points)]



    @staticmethod
    def evaluate_function(function: CompilableFunction, plot_points: jnp.ndarray) -> jnp.ndarray:
        compiled_function: CompiledFunction = function.compile(len(plot_points), plot_points.dtype)
        return compiled_function.evaluate(plot_points)



    @staticmethod
    def replace_nan_with_inf(array: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(jnp.isnan(array), jnp.inf, array)



    @staticmethod
    def clamp_function_values(function_values: jnp.ndarray, y_limits: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        clamped_values = jnp.where(function_values < y_limits[0], -jnp.inf, function_values)
        clamped_values = jnp.where(clamped_values > y_limits[1], jnp.inf, clamped_values)
        return clamped_values



    @staticmethod
    def adaptive_scatter_size(num_points: int, modifier: float = 1.0, min_size: float = 10.0) -> float:
        """Berechnet eine adaptive Scatter-Größe, die mit Anzahl der Punkte abnimmt, aber nicht kleiner als min_size wird."""
        base_size = 200 * modifier
        size = base_size / (num_points ** 0.5)  # Quadratwurzel-Abnahme: visuell angenehm
        return max(size, min_size)



    @staticmethod
    def scatter_size_for_equidistant_circles(number_of_circles: int, length_of_axis_inch: float) -> float:
        length_of_axis_pt: float = length_of_axis_inch * 72
        scatter_diameter_pt: float = length_of_axis_pt / number_of_circles

        return math.pi *  (scatter_diameter_pt / 2) ** 2



    @staticmethod
    def scatter_distance_for_equidistant_circles(number_of_circles: int, length_of_axis_coords: jnp.ndarray) -> jnp.ndarray:
        return length_of_axis_coords / number_of_circles



    @staticmethod
    def scatter_touch_distance(x_min: float, x_max: float, y_min: float, y_max: float, scatter_size: float, fig_width=6, fig_height=4, dpi=100):
        """
        Berechne den minimalen Abstand in Datenkoordinaten,
        den zwei Scatter-Punkte (Kreise) haben müssen, um sich exakt zu berühren.

        Parameter:
            x_min, x_max, y_min, y_max : float
                Achsenlimits (Datenbereich)
            scatter_size : float
                Scatter-Parameter (Markergröße in pt^2)
            fig_width, fig_height : float
                Größe der Figur in Zoll (Standard: 6x4)
            dpi : int
                Auflösung der Figur in dpi (Standard: 100)

        Rückgabe:
            (dx, dy) : tuple of float
                Radius in x- und y-Datenkoordinaten
                Die Distanz zweier Punkte zum Berühren ist dann 2*dx bzw. 2*dy
                (abhängig davon, ob man sie horizontal oder vertikal betrachtet).
        """
        # Radius in Punkten
        r_pt = math.sqrt(scatter_size / math.pi)
        # Umrechnung: pt -> inch
        r_inch = r_pt / 72.0

        # Größe der Achsen in Zoll
        ax_width_inch = fig_width
        ax_height_inch = fig_height

        # Datenbereich
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Daten pro Zoll
        x_per_inch = x_range / ax_width_inch
        y_per_inch = y_range / ax_height_inch

        # Radius in Datenkoordinaten
        dx = 2.75 * r_inch * x_per_inch
        dy = 2.75 * r_inch * y_per_inch

        return dx, dy



    @classmethod
    def create_plot_data_from_function_values(cls, function_name: str, function_index: int, plot_points: jnp.ndarray,
                    function_values: jnp.ndarray, y_limits: tuple[jnp.ndarray, jnp.ndarray]) -> FunctionPlotData:

        cleaned_function_values = cls.replace_nan_with_inf(function_values)
        clamped_function_values = cls.clamp_function_values(cleaned_function_values, y_limits)

        connectable_segments, single_points = cls._extract_connectable_segments_and_single_points_(
            function_index, plot_points, clamped_function_values, y_limits
        )

        return FunctionPlotData(function_name, function_index, connectable_segments, single_points)



    @classmethod
    def create_plot_data_from_compilable_function(cls, function: CompilableFunction, function_index: int,
                    plot_points: jnp.ndarray, y_limits: tuple[jnp.ndarray, jnp.ndarray]) -> FunctionPlotData:

        compiled_function: CompiledFunction = function.compile(len(plot_points), plot_points.dtype)
        function_values: jnp.ndarray = compiled_function.evaluate(plot_points)

        return PlotUtils.create_plot_data_from_function_values(function.name, function_index, plot_points,
                                                               function_values, y_limits)



    @staticmethod
    def create_plot_meta_info_str(pipeline_data: list[PipelineData], additional_data: AdditionalComponentExecutionData) -> str:
        pipeline_conf_name: str = additional_data.pipeline_configuration.name or ""
        pipeline_input_name: str = additional_data.pipeline_input.name or ""

        pipeline_conf_str: str = f"Conf.: {repr(pipeline_conf_name)}" if pipeline_conf_name else ""
        pipeline_input_str: str = f"Input: {repr(pipeline_input_name)}" if pipeline_input_name else ""

        data_types_str: set[str] = { data.data_type.__module__ + "." + data.data_type.__name__ for data in pipeline_data }
        node_counts_str: set[str] = { str(data.node_count) for data in pipeline_data }
        intervals_str: set[str] = { str(data.interpolation_interval) for data in pipeline_data }
        #node_count_str: str = str(pipeline_data[0].node_count)
        #interval_str: str = str(pipeline_data[0].interpolation_interval)

        return (f"{pipeline_conf_str}; {pipeline_input_str}; Dtypes: {data_types_str}; Node_counts: {node_counts_str}; "
                f"Intervals: {intervals_str}")



    # @classmethod
    # def _split_up_function_values_(cls, x_array: jnp.ndarray, function_values: jnp.ndarray) -> tuple[
    #     list[jnp.ndarray], list[jnp.ndarray], list[int]]:
    #     indices_list, inf_indices = cls._get_indices_of_piecewise_intervals_(function_values)
    #     nodes_list = []
    #     values_list = []
    #
    #     for indices in indices_list:
    #         nodes = x_array[indices[0]:indices[-1] + 1]
    #         nodes_list.append(nodes)
    #
    #         values = function_values[indices[0]:indices[-1] + 1]
    #         values_list.append(values)
    #
    #     return nodes_list, values_list, inf_indices


    @classmethod
    def _extract_connectable_segments_and_single_points_(cls, function_index: int, plot_points: jnp.ndarray,
                function_values: jnp.ndarray, y_limits: tuple[jnp.ndarray, jnp.ndarray]) \
                -> tuple[list[list[tuple[jnp.ndarray, jnp.ndarray]]], list[tuple[jnp.ndarray, jnp.ndarray]]]:

        connectable_indices_segments, single_point_indices = cls._extract_connectable_segments_and_single_points_indices_(function_values)

        connectable_segments = []
        single_points = []

        for segment_indices in connectable_indices_segments:
            segment: list[tuple[jnp.ndarray, jnp.ndarray]] = [ (plot_points[index], function_values[index]) for index in segment_indices ]
            connectable_segments.append(segment)

        for single_point_index in single_point_indices:
            old_y_value: jnp.ndarray = function_values[single_point_index]
            y_offset: float = function_index * 2 * math.sqrt(PlotUtils.adaptive_scatter_size(len(plot_points)) / math.pi)
            new_y_value: jnp.ndarray = y_limits[0] - y_offset if jnp.isneginf(old_y_value) else y_limits[1] + y_offset
            single_points.append((plot_points[single_point_index], new_y_value))

        return connectable_segments, single_points



    @staticmethod
    def _extract_connectable_segments_and_single_points_indices_(function_values: jnp.ndarray) -> tuple[list[list[int]], list[int]]:
        single_point_indices = []

        connectable_indices_segments = []
        connectable_indices = []

        for i, value in enumerate(function_values):
            if jnp.isfinite(value):
                connectable_indices.append(i)
            else:
                single_point_indices.append(i)

                if len(connectable_indices) > 0:
                    connectable_indices_segments.append(connectable_indices)
                    connectable_indices = []

        if len(connectable_indices) > 0:
            connectable_indices_segments.append(connectable_indices)

        return connectable_indices_segments, single_point_indices




# s: float = 600
#
# plt.figure(figsize=(6, 4))
#
# x_factor = 2
# y_factor = 3
#
#
#
#
# plt.scatter(0, 0, s=s)
# plt.scatter(x_factor, y_factor, s=s)
# plt.scatter(x_factor, -y_factor, s=s)
# plt.scatter(-x_factor, y_factor, s=s)
# plt.scatter(-x_factor, -y_factor, s=s)
#
# dx, dy = PlotUtils.scatter_touch_distance(-x_factor, x_factor, -y_factor, y_factor, s)
#
# plt.scatter(2.75 * dx, 0, s=s)
# plt.scatter(0, 2.75 * dy, s=s)
#
# # plt.yticks([0.5 * i - 1.5 for i in range(7)])
# # plt.xticks([0.5 * i - 1.5 for i in range(7)])
#
# plt.show()