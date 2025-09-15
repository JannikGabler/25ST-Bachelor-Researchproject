import math

import jax.numpy as jnp

from dataclasses import dataclass
from fractions import Fraction

from matplotlib.lines import Line2D

from constants.internal_logic_constants import BaseRoundOffErrorPlotComponentConstants
from data_classes.pipeline_data.pipeline_data import PipelineData
from data_classes.plot_template.plot_template import PlotTemplate
from data_classes.plotting.base_round_off_error_plot_component_utils_data.base_round_off_error_plot_component_utils_data import \
    BaseRoundOffErrorPlotComponentUtilsData
from exceptions.not_instantiable_error import NotInstantiableError
from functions.abstracts.compilable_function import CompilableFunction
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import \
    AdditionalComponentExecutionData
from utils.base_round_off_error_plot_component_utils import BaseRoundOffErrorPlotComponentUtils
from utils.plot_utils import PlotUtils


class RelativeRoundOffErrorPlotComponentUtils:
    """
    TODO
    """

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
        return BaseRoundOffErrorPlotComponentUtils.plot_data(pipeline_data, additional_data, cls._set_relative_round_off_errors_)



    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _set_relative_round_off_errors_(data: BaseRoundOffErrorPlotComponentUtilsData, pipeline_data: list[PipelineData]) -> None:
        data.round_off_errors = []
        eps = jnp.finfo(pipeline_data[0].data_type).eps

        for i, pd in enumerate(pipeline_data):
            function_values: jnp.ndarray = PlotUtils.evaluate_function(pd.interpolant, data.evaluation_points)
            interpolant_values_float: list[float] = [float(value) for value in data.interpolant_values_exact]
            interpolant_value_jnp: jnp.ndarray = jnp.array(interpolant_values_float)

            abs_round_off_errors: jnp.ndarray = jnp.abs(function_values - interpolant_value_jnp)
            rel_round_off_errors: jnp.ndarray = abs_round_off_errors / (interpolant_value_jnp + eps)
            data.round_off_errors.append(rel_round_off_errors)