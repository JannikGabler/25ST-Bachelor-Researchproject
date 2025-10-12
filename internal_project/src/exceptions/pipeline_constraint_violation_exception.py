from __future__ import annotations
from typing import TYPE_CHECKING

from general_data_structures.directional_acyclic_graph.directional_acyclic_graph_node import (
    DirectionalAcyclicGraphNode,
)

if TYPE_CHECKING:
    from pipeline_entities.pipeline_execution.dataclasses.pipeline_component_instantiation_info import (
        PipelineComponentInstantiationInfo,
    )
    from pipeline_entities.pipeline.component_entities.constraints.abstracts.constraint import (
        Constraint,
    )


class PipelineConstraintViolationException(Exception):
    _causing_node_: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]
    _causing_constraint_: Constraint

    def __init__(
        self,
        msg: str,
        causing_node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
        causing_constraint: Constraint,
    ) -> None:
        super().__init__(msg)

        self._causing_node_ = causing_node
        self._causing_constraint_ = causing_constraint

    @property
    def causing_node(
        self,
    ) -> DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]:
        return self._causing_node_

    @property
    def causing_constraint(self) -> Constraint:
        return self._causing_constraint_
