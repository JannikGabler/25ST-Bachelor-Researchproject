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
        self._attribute_name_ = attribute_name



    ######################
    ### Public methods ###
    ######################
    def evaluate(self, own_node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                 pipeline_configuration: PipelineConfiguration) -> bool:

        return self._is_attribute_for_component_guaranteed_set_(own_node)



    def get_error_message(self) -> str | None:
        return "TODO" # TODO



    ##########################
    ### Overridden methods ###
    ##########################
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(attribute_name={repr(self._attribute_name_)})"

    def __str__(self):
        return self.__repr__()



    def __hash__(self):
        return hash(self._attribute_name_)



    def __eq__(self, other):
        if not isinstance(other, self.__class__):   # Covers None
            return False
        else:
            return self._attribute_name_ == other._attribute_name_



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