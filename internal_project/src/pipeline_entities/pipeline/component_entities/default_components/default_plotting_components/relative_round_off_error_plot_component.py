from pipeline_entities.pipeline.component_entities.default_component_types.plot_component import PlotComponent
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.plot_components.relative_round_off_error_plot_component import relative_round_off_error_plot_component_meta_info
from utils.relative_round_off_error_plot_component_utils import RelativeRoundOffErrorPlotComponentUtils
from constants.internal_logic_constants import RelativeRoundOffErrorPlotComponentConstants


@pipeline_component(id="relative round off error plotter", type=PlotComponent, meta_info=relative_round_off_error_plot_component_meta_info)
class RelativeRoundOffErrorPlotComponent(PlotComponent):
    """
    Pipeline component that generates the relative round off error plot.
    """

    PLOT_COMPONENT_UTILS_CLASS = RelativeRoundOffErrorPlotComponentUtils
    PLOT_COMPONENT_CONSTANTS_CLASS = RelativeRoundOffErrorPlotComponentConstants
