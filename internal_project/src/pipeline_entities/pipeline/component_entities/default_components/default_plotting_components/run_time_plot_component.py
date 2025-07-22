import matplotlib.pyplot as plt
import numpy as np
import re

from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.plot_components.interpolants_plot_component_meta_info import \
    plot_component_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component import PipelineComponent
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline_execution.output.pipeline_component_execution_report import PipelineComponentExecutionReport


@pipeline_component(id="runtime plotter", type=InterpolationCore, meta_info=plot_component_meta_info)
class RunTimePlotComponent(InterpolationCore):
    """
    Plots the initialization and execution time of each interpolation method as bar chart.
    """

    def perform_action(self) -> PipelineData:
        pass