from __future__ import annotations
from typing import TYPE_CHECKING

from data_structures.tree.tree_node import TreeNode

if TYPE_CHECKING:
    from pipeline_entities.constraints.abstracts.static_constraint import StaticConstraint
    from pipeline_entities.component_info.dataclasses.pipeline_component_info import PipelineComponentInfo


class PipelineConfigurationConstraintException(Exception):
    _causing_node_: TreeNode[PipelineComponentInfo]
    _causing_constraint_: StaticConstraint



    def __init__(self, msg: str, causing_node: TreeNode[PipelineComponentInfo], causing_constraint: StaticConstraint) -> None:
        super().__init__(msg)

        self._causing_node_ = causing_node
        self._causing_constraint_ = causing_constraint



    @property
    def causing_node(self) -> TreeNode[PipelineComponentInfo]:
        return self._causing_node_

    @property
    def causing_constraint(self) -> StaticConstraint:
        return self._causing_constraint_