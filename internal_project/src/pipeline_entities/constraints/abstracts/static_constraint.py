from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from data_structures.tree.tree_node import TreeNode
from pipeline_entities.constraints.abstracts.constraint import Constraint
from pipeline_entities.constraints.enums.constraint_type import ConstraintType

if TYPE_CHECKING:
    from pipeline_entities.component_info.dataclasses.pipeline_component_info import PipelineComponentInfo
    from pipeline_entities.pipeline_configuration.dataclasses.pipeline_configuration import PipelineConfiguration


class StaticConstraint(Constraint, ABC):
    ##########################
    ### Attributs of class ###
    ##########################
    __constraint_type__: ConstraintType = ConstraintType.STATIC



    ######################
    ### Public methods ###
    ######################
    @abstractmethod
    def evaluate(self, own_tree_node: TreeNode[PipelineComponentInfo], pipeline_configuration: PipelineConfiguration) -> bool:
        pass



