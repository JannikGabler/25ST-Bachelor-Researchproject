from pipeline_entities.pipeline.component_entities.default_component_types.plot_component import (
    PlotComponent,
)

from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import (
    pipeline_component,
)
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.plot_components.relative_error_plot_component_meta_info import (
    relative_error_plot_component_meta_info,
)

from utils.relative_error_plot_component_utils import RelativeErrorPlotComponentUtils
from constants.internal_logic_constants import RelativeErrorPlotComponentConstants


@pipeline_component(
    id="relative error plotter",
    type=PlotComponent,
    meta_info=relative_error_plot_component_meta_info,
)
class AbsoluteErrorPlotComponent(PlotComponent):
    PLOT_COMPONENT_UTILS_CLASS = RelativeErrorPlotComponentUtils
    PLOT_COMPONENT_CONSTANTS_CLASS = RelativeErrorPlotComponentConstants
