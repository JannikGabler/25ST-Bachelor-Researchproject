from general_data_structures.directed_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode

from pipeline_entities.pipeline.component_entities.constraints.abstracts.static_constraint import StaticConstraint
from pipeline_entities.pipeline_execution.dataclasses.pipeline_component_instantiation_info import \
    PipelineComponentInstantiationInfo
from pipeline_entities.large_data_classes.pipeline_configuration.pipeline_configuration import PipelineConfiguration


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



    ######################
    ### Public methods ###
    ######################
    def evaluate(self, own_node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                 pipeline_configuration: PipelineConfiguration) -> bool:

        return len(own_node.predecessors) <= self._max_amount_



    def get_error_message(self) -> str | None:
        return "TODO" # TODO



    ##########################
    ### Overridden methods ###
    ##########################
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_amount={repr(self._max_amount_)})"

    def __str__(self):
        return self.__repr__()



    def __hash__(self):
        return hash(self._max_amount_)



    def __eq__(self, other):
        if not isinstance(other, self.__class__):   # Covers None
            return False
        else:
            return self._max_amount_ == other._max_amount_









