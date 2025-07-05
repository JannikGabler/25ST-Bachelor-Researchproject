from data_structures.directed_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode

from pipeline_entities.constraints.abstracts.static_constraint import StaticConstraint
from pipeline_entities.pipeline_component_instantiation_info.pipeline_component_instantiation_info import \
    PipelineComponentInstantiationInfo
from pipeline_entities.pipeline_configuration.dataclasses.pipeline_configuration import PipelineConfiguration


class MaxPredecessorsConstraint(StaticConstraint):
    ##############################
    ### Attributs of instances ###
    ##############################
    __max_amount__: int



    ###################
    ### Constructor ###
    ###################
    def __init__(self, max_amount: int):
        self.__max_amount__ = max_amount



    ##########################
    ### Overridden methods ###
    ##########################
    def evaluate(self, own_node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                 pipeline_configuration: PipelineConfiguration) -> bool:

        return len(own_node.predecessors) <= self.__max_amount__



    def __eq__(self, other):
        if not isinstance(other, MaxPredecessorsConstraint):   # Covers None
            return False
        else:
            return self.__max_amount__ == other.__max_amount__



    def __hash__(self):
        return hash(self.__max_amount__)



    def __repr__(self) -> str:
        return f"MaxPredecessorsConstraint(max_amount='{self.__max_amount__}')"

