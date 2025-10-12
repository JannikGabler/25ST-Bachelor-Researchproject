from __future__ import annotations

from general_data_structures.directional_acyclic_graph.directional_acyclic_graph_node import (
    DirectionalAcyclicGraphNode,
)
from pipeline_entities.pipeline_execution.dataclasses.pipeline_component_instantiation_info import (
    PipelineComponentInstantiationInfo,
)


class PipelineExecutionAttributeUnmodifiedException(Exception):
    _causing_node_: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]
    _causing_attribute_name_: str

    def __init__(
        self,
        msg: str,
        causing_node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
        causing_attribute_name: str,
    ) -> None:

        super().__init__(msg)

        self._causing_node_ = causing_node
        self._causing_attribute_name_ = causing_attribute_name

    @property
    def causing_pipeline_component(
        self,
    ) -> DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]:
        return self._causing_node_

    @property
    def causing_attribute_name(self) -> str:
        return self._causing_attribute_name_
