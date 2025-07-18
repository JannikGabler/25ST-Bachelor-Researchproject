from dataclasses import fields

from general_data_structures.directed_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode
from pipeline_entities.pipeline.component_entities.constraints.abstracts.static_constraint import StaticConstraint
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline_execution.dataclasses.pipeline_component_instantiation_info import \
    PipelineComponentInstantiationInfo
from pipeline_entities.large_data_classes.pipeline_configuration.pipeline_configuration import PipelineConfiguration


class AttributeRequiredConstraint(StaticConstraint):
    ##############################
    ### Attributs of instances ###
    ##############################
    _attribute_name_: str



    ###################
    ### Constructor ###
    ###################
    def __init__(self, attribute_name: str):
        if not any(field.name == attribute_name for field in fields(PipelineData)):
            raise AttributeError(f"The class 'PipelineData' has no attribute with the name '{attribute_name}'")

        self._attribute_name_ = attribute_name



    ##########################
    ### Overridden methods ###
    ##########################
    def evaluate(self, own_node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                 pipeline_configuration: PipelineConfiguration) -> bool:

        return self._is_attribute_for_component_guaranteed_set_(own_node)



    def __eq__(self, other):
        if not isinstance(other, AttributeRequiredConstraint):   # Covers None
            return False
        else:
            return self._attribute_name_ == other._attribute_name_



    def __hash__(self):
        return hash(self._attribute_name_)



    def __repr__(self) -> str:
        return f"AttributRequiredConstraint<attribute_name='{self._attribute_name_}'>"



    #######################
    ### Private methods ###
    #######################
    def _is_attribute_for_component_guaranteed_set_(self, node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]) -> bool:
        attributes_modifying: set[str] = node.value.component.component_meta_info.attributes_modifying

        if self._attribute_name_ in attributes_modifying:
            return True
        if not node.predecessors:
            return False

        return all(self._is_attribute_for_component_guaranteed_set_(predecessor) for predecessor in node.predecessors)