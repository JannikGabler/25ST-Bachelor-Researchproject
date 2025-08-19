# from __future__ import annotations
# from abc import ABC, abstractmethod
# from typing import TYPE_CHECKING
#
# from data_structures.directional_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode
# from pipeline_entities.constraints.pipeline_component.constraint import Constraint
# from pipeline_entities.constraints.enums.constraint_type import ConstraintType
#
# if TYPE_CHECKING:
#     from pipeline_entities.pipeline_data_transfer.pipeline_data import PipelineData
#     from pipeline_entities.pipeline_configuration.dataclasses.pipeline_configuration import PipelineConfiguration
#     from pipeline_entities.pipeline_input.pipeline_input import PipelineInput
#     from pipeline_entities.pipeline_component_instantiation_info.pipeline_component_instantiation_info import \
#         PipelineComponentInstantiationInfo
#
#
# class MixedConstraint(Constraint, ABC):
#     ##########################
#     ### Attributs of class ###
#     ##########################
#     __constraint_type__: ConstraintType = ConstraintType.MIXED
#
#
#
#     ######################
#     ### Public methods ###
#     ######################
#     @abstractmethod
#     def evaluate(self, pipeline_data: PipelineData, pipeline_input: PipelineInput,
#                  own_node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
#                  pipeline_configuration: PipelineConfiguration) -> bool:
#         pass
#
#
