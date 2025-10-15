from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline_entities.pipeline.component_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo


@dataclass(frozen=True)
class PipelineComponentInfo:
    """
    Information object that describes a pipeline component.
    This dataclass bundles the identification, type, class implementation, and meta information of a pipeline component.

    Attributes:
        component_id (str): Unique identifier of the component.
        component_type (type): The general type of the component.
        component_class (type): The concrete class that implements the component.
        component_meta_info (ComponentMetaInfo): Meta information defining attribute modifications and constraints for the component.
    """

    component_id: str
    component_type: type
    component_class: type
    component_meta_info: ComponentMetaInfo
