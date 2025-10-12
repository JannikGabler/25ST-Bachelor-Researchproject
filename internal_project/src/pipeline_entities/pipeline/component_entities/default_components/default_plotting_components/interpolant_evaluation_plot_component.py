from pipeline_entities.pipeline.component_entities.default_component_types.plot_component import (
    PlotComponent,
)

from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import (
    pipeline_component,
)
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.plot_components.interpolant_evaluation_plot_component_meta_info import (
    interpolant_evaluation_plot_component_meta_info,
)

from constants.internal_logic_constants import OldInterpolantsPlotComponentConstants
from utils.interpolant_evaluation_plot_component_utils import (
    InterpolantEvaluationPlotComponentUtils,
)


@pipeline_component(
    id="interpolant evaluation plotter",
    type=PlotComponent,
    meta_info=interpolant_evaluation_plot_component_meta_info,
)
class InterpolantEvaluationPlotComponent(PlotComponent):
    PLOT_COMPONENT_UTILS_CLASS = InterpolantEvaluationPlotComponentUtils
    PLOT_COMPONENT_CONSTANTS_CLASS = OldInterpolantsPlotComponentConstants
