from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline_entities.pipeline.component_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo


@dataclass(frozen=True)
class PipelineComponentInfo:
    component_id: str
    component_type: type
    component_class: type
    component_meta_info: ComponentMetaInfo


