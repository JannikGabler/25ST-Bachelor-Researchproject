from data_structures.directed_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode

from pipeline_entities.constraints.abstracts.static_constraint import StaticConstraint
from pipeline_entities.pipeline_component_instantiation_info.pipeline_component_instantiation_info import \
    PipelineComponentInstantiationInfo
from pipeline_entities.pipeline_configuration.dataclasses.pipeline_configuration import PipelineConfiguration


class MaxPredecessorsConstraint(StaticConstraint):
    ##############################
    ### Attributs of instances ###
    ##############################
    _max_amount_: int



    ###################
    ### Constructor ###
    ###################
    def __init__(self, max_amount: int):
        self._max_amount_ = max_amount



    ##########################
    ### Overridden methods ###
    ##########################
    def evaluate(self, own_node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                 pipeline_configuration: PipelineConfiguration) -> bool:

        return len(own_node.predecessors) <= self._max_amount_



    def __eq__(self, other):
        if not isinstance(other, MaxPredecessorsConstraint):   # Covers None
            return False
        else:
            return self._max_amount_ == other._max_amount_



    def __hash__(self):
        return hash(self._max_amount_)



    def __repr__(self) -> str:
        return f"MaxPredecessorsConstraint(max_amount={self._max_amount_})"

