from dataclasses import dataclass
from typing import Any

from pipeline_entities.component_info.dataclasses.pipeline_component_info import PipelineComponentInfo


@dataclass
class PipelineComponentInstantiationInfo:
    component_name: str

    component: PipelineComponentInfo

    component_specific_arguments: dict[str, Any]
