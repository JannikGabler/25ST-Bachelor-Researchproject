from dataclasses import dataclass

from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline_execution.dataclasses.pipeline_component_instantiation_info import \
    PipelineComponentInstantiationInfo


@dataclass
class PipelineComponentExecutionReport:
    component_instantiation_info: PipelineComponentInstantiationInfo | None = None

    component_output: PipelineData | None = None

    component_init_time: float | None = None
    component_execution_time: float | None = None