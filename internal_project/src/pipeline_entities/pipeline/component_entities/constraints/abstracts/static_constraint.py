from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from general_data_structures.directional_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode
from pipeline_entities.pipeline.component_entities.constraints.abstracts.constraint import Constraint
from pipeline_entities.pipeline.component_entities.constraints.enums.constraint_type import ConstraintType

if TYPE_CHECKING:
    from data_classes.pipeline_configuration.pipeline_configuration import PipelineConfiguration
    from pipeline_entities.pipeline_execution.dataclasses.pipeline_component_instantiation_info import \
        PipelineComponentInstantiationInfo


class StaticConstraint(Constraint, ABC):
    ##########################
    ### Attributs of class ###
    ##########################
    __constraint_type__: ConstraintType = ConstraintType.STATIC



    ######################
    ### Public methods ###
    ######################
    @abstractmethod
    def evaluate(self, own_node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo], pipeline_configuration: PipelineConfiguration) -> bool:
        pass



