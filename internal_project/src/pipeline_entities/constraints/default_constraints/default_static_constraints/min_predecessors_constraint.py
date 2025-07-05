from data_structures.directed_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode

from pipeline_entities.constraints.abstracts.static_constraint import StaticConstraint
from pipeline_entities.pipeline_component_instantiation_info.pipeline_component_instantiation_info import \
    PipelineComponentInstantiationInfo
from pipeline_entities.pipeline_configuration.dataclasses.pipeline_configuration import PipelineConfiguration


class MinPredecessorsConstraint(StaticConstraint):
    ##############################
    ### Attributs of instances ###
    ##############################
    __min_amount__: int



    ###################
    ### Constructor ###
    ###################
    def __init__(self, min_amount: int):
        self.__min_amount__ = min_amount



    ##########################
    ### Overridden methods ###
    ##########################
    def evaluate(self, own_node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                 pipeline_configuration: PipelineConfiguration) -> bool:

        return len(own_node.predecessors) >= self.__min_amount__



    def __eq__(self, other):
        if not isinstance(other, MinPredecessorsConstraint):   # Covers None
            return False
        else:
            return self.__min_amount__ == other.__min_amount__



    def __hash__(self):
        return hash(self.__min_amount__)



    def __repr__(self) -> str:
        return f"MinPredecessorsConstraint(min_amount='{self.__min_amount__}')"

