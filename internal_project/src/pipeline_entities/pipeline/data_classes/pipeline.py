from dataclasses import dataclass

from pipeline_entities.large_data_classes.pipeline_configuration.pipeline_configuration import PipelineConfiguration
from pipeline_entities.large_data_classes.pipeline_input.pipeline_input import PipelineInput


@dataclass(frozen=True)
class Pipeline:
    pipeline_configuration: PipelineConfiguration
    pipeline_input: PipelineInput



