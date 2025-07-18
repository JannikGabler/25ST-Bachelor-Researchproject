from general_data_structures.directed_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode

from pipeline_entities.pipeline.component_entities.constraints.abstracts.static_constraint import StaticConstraint
from pipeline_entities.pipeline_execution.dataclasses.pipeline_component_instantiation_info import \
    PipelineComponentInstantiationInfo
from pipeline_entities.large_data_classes.pipeline_configuration.pipeline_configuration import PipelineConfiguration


class MinPredecessorsConstraint(StaticConstraint):
    ##############################
    ### Attributs of instances ###
    ##############################
    _min_amount_: int



    ###################
    ### Constructor ###
    ###################
    def __init__(self, min_amount: int):
        self._min_amount_ = min_amount



    ##########################
    ### Overridden methods ###
    ##########################
    def evaluate(self, own_node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                 pipeline_configuration: PipelineConfiguration) -> bool:

        return len(own_node.predecessors) >= self._min_amount_



    def __eq__(self, other):
        if not isinstance(other, MinPredecessorsConstraint):   # Covers None
            return False
        else:
            return self._min_amount_ == other._min_amount_



    def __hash__(self):
        return hash(self._min_amount_)



    def __repr__(self) -> str:
        return f"MinPredecessorsConstraint(min_amount='{self._min_amount_}')"

