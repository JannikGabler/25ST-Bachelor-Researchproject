import math
from fractions import Fraction

import jax.numpy as jnp
from numpy.typing import DTypeLike

from exceptions.not_instantiable_error import NotInstantiableError
from functions.abstracts.compilable_function import CompilableFunction
from functions.abstracts.compiled_function import CompiledFunction
from data_classes.pipeline_data.pipeline_data import PipelineData
from data_classes.plotting_data.function_plot_data import FunctionPlotData
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData



class PlotUtils:
    """
    Utility helpers for plotting and plot-preparation tasks. This class is not meant to be instantiated.
    """

    ###################
    ### Constructor ###
    ###################
    def __init__(self) -> None:
        """
        Raises:
            NotInstantiableError: Always raised when initialized to indicate that this class is not meant to be instantiated.
        """

        raise NotInstantiableError(f"The class {repr(self.__class__.__name__)} cannot be instantiated.")


    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def create_plot_points(interval: jnp.ndarray, amount_of_evaluation_points: int) -> jnp.ndarray:
        """
        Create evenly spaced plot points within the given interval.

        Args:
            interval (jnp.ndarray): Interval where to plot.
            amount_of_evaluation_points: Number of points to create.

        Returns:
            jnp.ndarray: Equally spaced points over the interval.
        """

        return jnp.linspace(interval[0], interval[1], amount_of_evaluation_points, dtype=jnp.float32)


    @staticmethod
    def create_exact_plot_points(interval: jnp.ndarray, amount_of_evaluation_points: int) -> list[Fraction]:
        """
        Create evenly spaced plot points as exact rational Fractions within the given interval.

        Args:
            interval (jnp.ndarray): Interval defining the range of plot points.
            amount_of_evaluation_points (int): Number of points to generate.

        Returns:
            list[Fraction]: Equally spaced rational points across the interval.
        """

        left_end: Fraction = Fraction.from_float(interval[0].astype(float).item())
        right_end: Fraction = Fraction.from_float(interval[1].astype(float).item())

        step_size: Fraction = (right_end - left_end) / (amount_of_evaluation_points - 1)
        return [left_end + i * step_size for i in range(amount_of_evaluation_points)]


    @staticmethod
    def evaluate_function(function: CompilableFunction, data_type: DTypeLike, plot_points: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate a compiled function over the given plot points.

        Args:
            function (CompilableFunction): Function object that can be compiled for evaluation.
            data_type (DTypeLike): Data type to use for computation.
            plot_points (jnp.ndarray): Input points where the function will be evaluated.

        Returns:
            jnp.ndarray: Evaluated function values corresponding to the given plot points.
         """

        compiled_function: CompiledFunction = function.compile(len(plot_points), data_type)
        return compiled_function.evaluate(plot_points.astype(data_type))


    @staticmethod
    def replace_nan_with_inf(array: jnp.ndarray) -> jnp.ndarray:
        """
        Replace NaN values with inf.

        Args:
            array (jnp.ndarray): Input array.

        Returns:
            jnp.ndarray: Array with NaNs replaced by inf.
        """

        return jnp.where(jnp.isnan(array), jnp.inf, array)


    @staticmethod
    def clamp_function_values(function_values: jnp.ndarray, y_limits: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """
        Clamp values outside y-limits to ±inf for plotting.

        Args:
            function_values (jnp.ndarray): Values to clamp.
            y_limits (tuple[jnp.ndarray, jnp.ndarray]): (y_min, y_max) limits.

        Returns:
            jnp.ndarray: Values with out-of-range entries set to ±inf.
        """

        clamped_values = jnp.where(function_values < y_limits[0], -jnp.inf, function_values)
        clamped_values = jnp.where(clamped_values > y_limits[1], jnp.inf, clamped_values)
        return clamped_values

    @staticmethod
    def adaptive_scatter_size(num_points: int, modifier: float = 1.0, min_size: float = 10.0) -> float:
        """
        Compute an adaptive scatter size.

        Args:
            num_points (int): Number of scatter points.
            modifier (float): Scale factor for base size.
            min_size (float): Minimum allowed size.

        Returns:
            float: Scatter marker size.
        """

        base_size = 200 * modifier
        size = base_size / (num_points**0.5)  # Quadratwurzel-Abnahme: visuell angenehm
        return max(size, min_size)


    @staticmethod
    def scatter_size_for_equidistant_circles(number_of_circles: int, length_of_axis_inch: float) -> float:
        """
        Compute the scatter point size required to display a given number of equally spaced circles along an axis.

        Args:
            number_of_circles (int): Number of circles to distribute along the axis.
            length_of_axis_inch (float): Physical length of the axis in inches.

        Returns:
            float: Scatter point area in points² for equally spaced circles.
        """

        length_of_axis_pt: float = length_of_axis_inch * 72
        scatter_diameter_pt: float = length_of_axis_pt / number_of_circles

        return math.pi * (scatter_diameter_pt / 2) ** 2


    @staticmethod
    def scatter_distance_for_equidistant_circles(number_of_circles: int, length_of_axis_coords: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the coordinate distance between equally spaced circles along an axis.

        Args:
            number_of_circles (int): Number of circles to distribute along the axis.
            length_of_axis_coords (jnp.ndarray): Axis length expressed in coordinate units.

        Returns:
            jnp.ndarray: Distance between adjacent circles in coordinate units.
        """

        return length_of_axis_coords / number_of_circles


    @staticmethod
    def scatter_touch_distance(x_min: float, x_max: float, y_min: float, y_max: float, scatter_size: float, fig_width=6, fig_height=4, dpi=100):
        """
        Compute the minimal distance in data coordinates required for two scatter points (circles) to exactly touch each other.

        Args:
            x_min (float): Minimum x-axis value of the data range.
            x_max (float): Maximum x-axis value of the data range.
            y_min (float): Minimum y-axis value of the data range.
            y_max (float): Maximum y-axis value of the data range.
            scatter_size (float): Scatter marker size in points² (as used by matplotlib).
            fig_width (float): Width of the figure in inches. Defaults to 6.
            fig_height (float): Height of the figure in inches. Defaults to 4.
            dpi (int): Figure resolution in dots per inch. Defaults to 100.

        Returns:
            tuple[float, float]: The (dx, dy) radii in x and y data coordinates.
            The distance for two circles to touch equals 2 * dx or 2 * dy, depending on the orientation.
        """

        r_pt = math.sqrt(scatter_size / math.pi)
        r_inch = r_pt / 72.0

        ax_width_inch = fig_width
        ax_height_inch = fig_height

        x_range = x_max - x_min
        y_range = y_max - y_min

        x_per_inch = x_range / ax_width_inch
        y_per_inch = y_range / ax_height_inch

        dx = 2.75 * r_inch * x_per_inch
        dy = 2.75 * r_inch * y_per_inch

        return dx, dy


    @classmethod
    def create_plot_data_from_function_values(cls, function_name: str, function_index: int, plot_points: jnp.ndarray, function_values: jnp.ndarray, y_limits: tuple[jnp.ndarray, jnp.ndarray]) -> FunctionPlotData:
        """
        Build plot data from precomputed function values.

        Args:
            function_name (str): Function name.
            function_index (int): Function index.
            plot_points (jnp.ndarray): Plot points.
            function_values (jnp.ndarray): Function values.
            y_limits (tuple[jnp.ndarray, jnp.ndarray]): (y_min, y_max) limits.

        Returns:
                FunctionPlotData: Segments and single points ready for plotting.
        """

        cleaned_function_values = cls.replace_nan_with_inf(function_values)
        clamped_function_values = cls.clamp_function_values(cleaned_function_values, y_limits)

        connectable_segments, single_points = (cls._extract_connectable_segments_and_single_points_(function_index, plot_points, clamped_function_values, y_limits))

        return FunctionPlotData(function_name, function_index, connectable_segments, single_points)


    @classmethod
    def create_plot_data_from_compilable_function(cls, function: CompilableFunction, function_index: int, plot_points: jnp.ndarray, y_limits: tuple[jnp.ndarray, jnp.ndarray]) -> FunctionPlotData:
        """
        Compile and evaluate a function on plot points and build plot data.

        Args:
            function (CompilableFunction): Function to compile.
            function_index (int): Function index.
            plot_points (jnp.ndarray): Plot points.
            y_limits (tuple[jnp.ndarray, jnp.ndarray]): (y_min, y_max) limits.

        Returns:
            FunctionPlotData: Segments and single points ready for plotting.
        """

        compiled_function: CompiledFunction = function.compile(len(plot_points), plot_points.dtype)
        function_values: jnp.ndarray = compiled_function.evaluate(plot_points)

        return PlotUtils.create_plot_data_from_function_values(function.name, function_index, plot_points, function_values, y_limits)


    @staticmethod
    def create_plot_meta_info_str(pipeline_data: list[PipelineData], additional_data: AdditionalComponentExecutionData) -> str:
        """
        Create a compact meta-info string for plot subtitles.

        Args:
            pipeline_data (list[PipelineData]): List of pipeline data.
            additional_data (AdditionalComponentExecutionData): Additional execution data.

        Returns:
            str: Meta information string (config, input, dtypes, node counts, intervals).
        """

        pipeline_conf_name: str = additional_data.pipeline_configuration.name or ""
        pipeline_input_name: str = additional_data.pipeline_input.name or ""

        pipeline_conf_str: str = (f"Conf.: {repr(pipeline_conf_name)}" if pipeline_conf_name else "")
        pipeline_input_str: str = (f"Input: {repr(pipeline_input_name)}" if pipeline_input_name else "")

        data_types_str: set[str] = {data.data_type.__module__ + "." + data.data_type.__name__ for data in pipeline_data}
        node_counts_str: set[str] = {str(data.node_count) for data in pipeline_data}
        intervals_str: set[str] = {str(data.interpolation_interval) for data in pipeline_data}

        return f"{pipeline_conf_str}; {pipeline_input_str}; Dtypes: {data_types_str}; Node_counts: {node_counts_str}; "f"Intervals: {intervals_str}"


    @classmethod
    def _extract_connectable_segments_and_single_points_(cls, function_index: int, plot_points: jnp.ndarray, function_values: jnp.ndarray, y_limits: tuple[jnp.ndarray, jnp.ndarray]) \
        -> tuple[list[list[tuple[jnp.ndarray, jnp.ndarray]]], list[tuple[jnp.ndarray, jnp.ndarray]]]:

        connectable_indices_segments, single_point_indices = (cls._extract_connectable_segments_and_single_points_indices_(function_values))

        connectable_segments = []
        single_points = []

        for segment_indices in connectable_indices_segments:
            segment: list[tuple[jnp.ndarray, jnp.ndarray]] = [(plot_points[index], function_values[index]) for index in segment_indices]
            connectable_segments.append(segment)

        for single_point_index in single_point_indices:
            old_y_value: jnp.ndarray = function_values[single_point_index]
            y_offset: float = (function_index * 2 * math.sqrt(PlotUtils.adaptive_scatter_size(len(plot_points)) / math.pi))
            new_y_value: jnp.ndarray = (y_limits[0] - y_offset if jnp.isneginf(old_y_value) else y_limits[1] + y_offset)
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
