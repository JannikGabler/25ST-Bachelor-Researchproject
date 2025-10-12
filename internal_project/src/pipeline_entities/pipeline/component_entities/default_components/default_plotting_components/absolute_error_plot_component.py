from pipeline_entities.pipeline.component_entities.default_component_types.plot_component import (
    PlotComponent,
)

from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import (
    pipeline_component,
)
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.plot_components.absolute_error_plot_component_meta_info import (
    absolut_error_plot_component_meta_info,
)

from utils.absolute_error_plot_component_utils import AbsoluteErrorPlotComponentUtils
from constants.internal_logic_constants import AbsoluteErrorPlotComponentConstants


@pipeline_component(
    id="absolute error plotter",
    type=PlotComponent,
    meta_info=absolut_error_plot_component_meta_info,
)
class AbsoluteErrorPlotComponent(PlotComponent):
    PLOT_COMPONENT_UTILS_CLASS = AbsoluteErrorPlotComponentUtils
    PLOT_COMPONENT_CONSTANTS_CLASS = AbsoluteErrorPlotComponentConstants
