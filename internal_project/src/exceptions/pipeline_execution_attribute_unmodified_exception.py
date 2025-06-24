from __future__ import annotations
from typing import TYPE_CHECKING

from data_structures.tree.tree_node import TreeNode

if TYPE_CHECKING:
    from pipeline_entities.component_info.dataclasses.pipeline_component_info import PipelineComponentInfo


class PipelineExecutionAttributeUnmodifiedException(Exception):
    _causing_node_: TreeNode[PipelineComponentInfo]
    _causing_attribute_name_: str



    def __init__(self, msg: str, causing_node: TreeNode[PipelineComponentInfo], causing_attribute_name: str) -> None:
        super().__init__(msg)

        self._causing_node_ = causing_node
        self._causing_attribute_name_ = causing_attribute_name



    @property
    def causing_pipeline_component(self) -> TreeNode[PipelineComponentInfo]:
        return self._causing_node_

    @property
    def causing_attribute_name(self) -> str:
        return self._causing_attribute_name_
