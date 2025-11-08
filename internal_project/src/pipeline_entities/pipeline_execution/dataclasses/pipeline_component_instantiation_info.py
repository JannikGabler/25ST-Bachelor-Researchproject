from dataclasses import dataclass
from typing import Any

from pipeline_entities.pipeline.component_entities.component_info.dataclasses.pipeline_component_info import (
    PipelineComponentInfo,
)


@dataclass
class PipelineComponentInstantiationInfo:
    component_name: str

    component: PipelineComponentInfo

    overridden_attributes: dict[str, Any]
