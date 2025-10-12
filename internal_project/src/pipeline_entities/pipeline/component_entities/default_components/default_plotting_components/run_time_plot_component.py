import matplotlib.pyplot as plt
import numpy as np
import re

from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.plot_components.run_time_plot_component_meta_info import (
    run_time_component_meta_info,
)
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import (
    InterpolationCore,
)
from pipeline_entities.pipeline.component_entities.default_component_types.plot_component import (
    PlotComponent,
)
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import (
    pipeline_component,
)
from data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import (
    AdditionalComponentExecutionData,
)


@pipeline_component(
    id="runtime plotter", type=PlotComponent, meta_info=run_time_component_meta_info
)
class RunTimePlotComponent(InterpolationCore):
    """
    Plots the initialization and execution time of each interpolation method as bar chart.
    """

    # TODO change to match parent class => enable concurrent plots and storing plots

    def __init__(
        self,
        pipeline_data: list[PipelineData],
        additional_execution_data: AdditionalComponentExecutionData,
    ):
        super().__init__(pipeline_data, additional_execution_data)
        self._additional_execution_data = additional_execution_data
        self._pipeline_data = pipeline_data

    def perform_action(self) -> PipelineData:
        execution_reports = list(
            self._additional_execution_data.component_execution_reports.values()
        )

        interpolant_reports = [
            report
            for report in execution_reports
            if issubclass(
                report.component_instantiation_info.component.component_type,
                InterpolationCore,
            )
            and report.component_instantiation_info.component.component_id.lower()
            != "run time plotter"
        ]

        method_names = []
        init_times = []
        exec_times = []

        for report in interpolant_reports:
            component_cls = (
                report.component_instantiation_info.component.component_class
            )
            raw_name = component_cls.__name__
            for suffix in ["Interpolant", "InterpolationCore"]:
                if raw_name.endswith(suffix):
                    raw_name = raw_name[: -len(suffix)]
            pretty_name = re.sub(r"(?<!^)(?=[A-Z])", " ", raw_name)

            method_names.append(pretty_name)
            init_times.append(report.component_init_time)
            exec_times.append(report.average_component_execution_time)

        x = np.arange(len(method_names))
        width = 0.5

        # Plot for init time
        plt.figure(figsize=(12, 6))
        plt.bar(x, init_times, width, color="skyblue")
        plt.xlabel("Interpolation Methods")
        plt.ylabel("Init Time (s)")
        plt.title("Initialization Time of the Interpolation Methods")
        plt.xticks(ticks=x, labels=method_names, rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show(block=False)

        # Plot for execution time
        plt.figure(figsize=(12, 6))
        plt.bar(x, exec_times, width, color="steelblue")
        plt.xlabel("Interpolation Methods")
        plt.ylabel("Execution Time (s)")
        plt.title("Execution Time of the Interpolation Methods")
        plt.xticks(ticks=x, labels=method_names, rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show(block=False)

        # print(self._pipeline_data_[0].__dict__)
        return self._pipeline_data[0]
