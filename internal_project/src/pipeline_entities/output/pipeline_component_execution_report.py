from dataclasses import dataclass

from pipeline_entities.data_transfer.pipeline_data import PipelineData
from pipeline_entities.pipeline_component_instantiation_info.pipeline_component_instantiation_info import \
    PipelineComponentInstantiationInfo


@dataclass
class PipelineComponentExecutionReport:
    component_instantiation_info: PipelineComponentInstantiationInfo | None = None

    component_output: PipelineData | None = None

    component_init_time: float | None = None
    component_execution_time: float | None = None