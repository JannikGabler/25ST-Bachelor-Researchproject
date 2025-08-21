from __future__ import annotations

import jax.numpy as jnp

from dataclasses import dataclass

from constants.internal_logic_constants import InterpolantsPlotComponentConstants
from functions.abstracts.compiled_function import CompiledFunction
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.large_data_classes.plotting_data.function_plot_data import FunctionPlotData
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import \
    AdditionalComponentExecutionData
from utils.plot_utils import PlotUtils


@dataclass
class InterpolantsPlotComponentData:
    meta_info_str: str

    plot_points: jnp.ndarray

    interpolation_nodes: jnp.ndarray
    interpolation_values: jnp.ndarray

    original_function_plot_data: FunctionPlotData
    interpolants_plot_data: list[FunctionPlotData]



    @classmethod
    def create_from(cls, pipeline_data: list[PipelineData],
                    additional_execution_info: AdditionalComponentExecutionData) -> InterpolantsPlotComponentData:
        main_data: PipelineData = pipeline_data[0]
        interval: jnp.ndarray = main_data.interpolation_interval

        plot_points: jnp.ndarray = PlotUtils.create_plot_points(interval, InterpolantsPlotComponentConstants.AMOUNT_OF_EVALUATION_POINTS)

        original_function: callable = main_data.original_function
        original_function_values: jnp.ndarray = original_function(plot_points)
        original_function_values = PlotUtils.replace_nan_with_inf(original_function_values)

        y_limits: tuple[jnp.ndarray, jnp.ndarray] = cls._calc_limits_(original_function_values)

        meta_info_str: str = PlotUtils.create_plot_meta_info_str(pipeline_data, additional_execution_info)
        original_function_plot_data: FunctionPlotData = cls._generate_plot_data_for_original_functions_(plot_points,
                                                                                                        original_function_values,
                                                                                                        y_limits)

        interpolants_plot_data: list[FunctionPlotData] = cls._generate_plot_data_for_interpolant_(plot_points,
                                                                                                  pipeline_data,
                                                                                                  y_limits)

        return InterpolantsPlotComponentData(meta_info_str, plot_points, main_data.interpolation_nodes,
                                             main_data.interpolation_values, original_function_plot_data,
                                             interpolants_plot_data)



    @classmethod
    def _calc_limits_(cls, original_function_values: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        valid_mask: jnp.ndarray = jnp.isfinite(original_function_values)

        clean_values = original_function_values[valid_mask]
        min_value = jnp.min(clean_values)
        max_value = jnp.max(clean_values)
        difference = InterpolantsPlotComponentConstants.Y_LIMIT_FACTOR * (max_value - min_value)
        return min_value - difference, max_value + difference



    @classmethod
    def _generate_plot_data_for_original_functions_(cls, plot_points: jnp.ndarray, original_function_values,
                                                    y_limits: tuple[jnp.ndarray, jnp.ndarray]) -> FunctionPlotData:
        return PlotUtils.create_plot_data("Original function", 0, plot_points,
                                          original_function_values, y_limits)



    @classmethod
    def _generate_plot_data_for_interpolant_(cls, plot_points: jnp.ndarray, pipeline_data: list[PipelineData],
                                             y_limits: tuple[jnp.ndarray, jnp.ndarray]) -> list[FunctionPlotData]:
        plot_data: list[FunctionPlotData] = []

        for i, data in enumerate(pipeline_data):
            interpolant: CompiledFunction = data.interpolant.compile(len(plot_points), data.data_type)
            function_values: jnp.ndarray = interpolant.evaluate(plot_points.astype(data.data_type))
            function_values = PlotUtils.replace_nan_with_inf(function_values)
            function_values = PlotUtils.clamp_function_values(function_values, y_limits)

            plot_data.append(PlotUtils.create_plot_data(
                data.interpolant.name, i+1, plot_points, function_values, y_limits
            ))

        return plot_data