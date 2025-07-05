from __future__ import annotations
from typing import TYPE_CHECKING

from data_structures.directed_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode
from data_structures.tree.tree_node import TreeNode
from pipeline_entities.pipeline_component_instantiation_info.pipeline_component_instantiation_info import \
    PipelineComponentInstantiationInfo


class PipelineExecutionAttributeModifiedException(Exception):
    _causing_node_: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]
    _causing_attribute_name_: str



    def __init__(self, msg: str, causing_node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                 causing_attribute_name: str) -> None:

        super().__init__(msg)

        self._causing_node_ = causing_node
        self._causing_attribute_name_ = causing_attribute_name



    @property
    def causing_pipeline_component(self) -> DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]:
        return self._causing_node_

    @property
    def causing_attribute_name(self) -> str:
        return self._causing_attribute_name_
