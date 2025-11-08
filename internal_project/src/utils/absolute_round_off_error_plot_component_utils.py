import jax.numpy as jnp

from data_classes.pipeline_data.pipeline_data import PipelineData
from data_classes.plot_template.plot_template import PlotTemplate
from data_classes.plotting.base_round_off_error_plot_component_utils_data.base_round_off_error_plot_component_utils_data import BaseRoundOffErrorPlotComponentUtilsData
from exceptions.not_instantiable_error import NotInstantiableError
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from utils.base_round_off_error_plot_component_utils import BaseRoundOffErrorPlotComponentUtils
from utils.plot_utils import PlotUtils


class AbsoluteRoundOffErrorPlotComponentUtils:
    """
    Utility helpers for generating absolute round-off error plots. This class cannot be instantiated.
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
    def plot_data(cls, pipeline_data: list[PipelineData], additional_data: AdditionalComponentExecutionData) -> PlotTemplate:
        """
        Create the absolute round-off error plot for the given pipeline data.

        Args:
            pipeline_data (list[PipelineData]): Data containing interpolant functions to evaluate.
            additional_data (AdditionalComponentExecutionData): Additional information for plot annotation.

        Returns:
            PlotTemplate: Configured plot visualizing the absolute round-off errors.
        """

        template: PlotTemplate = BaseRoundOffErrorPlotComponentUtils.plot_data(pipeline_data, additional_data, cls._set_absolute_round_off_errors_)

        template.fig.suptitle(f"Absolute round-off error plot")
        template.ax.set_xlabel("$x$")
        template.ax.set_ylabel("$\Delta f(x)$")
        template.fig.tight_layout()

        return template


    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _set_absolute_round_off_errors_(data: BaseRoundOffErrorPlotComponentUtilsData, pipeline_data: list[PipelineData]) -> None:
        data.round_off_errors = []

        for i, pd in enumerate(pipeline_data):
            function_values: jnp.ndarray = PlotUtils.evaluate_function(pd.interpolant, pd.data_type, data.evaluation_points)
            cast_function_values: jnp.ndarray = function_values.astype(jnp.float32)

            interpolant_values_float: list[float] = [float(value) for value in data.interpolant_values_exact]
            cast_interpolant_values: jnp.ndarray = jnp.array(interpolant_values_float, dtype=jnp.float32)

            abs_round_off_errors: jnp.ndarray = jnp.abs(cast_function_values - cast_interpolant_values)
            data.round_off_errors.append(abs_round_off_errors)
