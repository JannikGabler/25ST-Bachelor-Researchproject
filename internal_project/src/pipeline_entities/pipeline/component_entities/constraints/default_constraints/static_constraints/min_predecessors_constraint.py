from general_data_structures.directional_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode

from pipeline_entities.pipeline.component_entities.constraints.abstracts.static_constraint import StaticConstraint
from pipeline_entities.pipeline_execution.dataclasses.pipeline_component_instantiation_info import \
    PipelineComponentInstantiationInfo
from pipeline_entities.large_data_classes.pipeline_configuration.pipeline_configuration import PipelineConfiguration


class MinPredecessorsConstraint(StaticConstraint):
    ##############################
    ### Attributs of instances ###
    ##############################
    _min_amount_: int
    _error_message_: str | None



    ###################
    ### Constructor ###
    ###################
    def __init__(self, min_amount: int):
        self._min_amount_ = min_amount
        self._error_message_ = None



    ######################
    ### Public methods ###
    ######################
    def evaluate(self, own_node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                 pipeline_configuration: PipelineConfiguration) -> bool:

        if len(own_node.predecessors) >= self._min_amount_:
            self._error_message_ = None
            return True
        else:
            self._error_message_ = (
                f"Too few predecessors: found {len(own_node.predecessors)}, "
                f"but the minimum required number is {self._min_amount_}."
            )
            return False



    def get_error_message(self) -> str | None:
        return self._error_message_



    ##########################
    ### Overridden methods ###
    ##########################
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_amount={repr(self._min_amount_)})"

    def __str__(self):
        return self.__repr__()



    def __hash__(self):
        return hash(self._min_amount_)



    def __eq__(self, other):
        if not isinstance(other, self.__class__):   # Covers None
            return False
        else:
            return self._min_amount_ == other._min_amount_









