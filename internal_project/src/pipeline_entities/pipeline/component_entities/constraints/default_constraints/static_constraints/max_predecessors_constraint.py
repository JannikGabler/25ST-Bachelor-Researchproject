from general_data_structures.directional_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode

from pipeline_entities.pipeline.component_entities.constraints.abstracts.static_constraint import StaticConstraint
from pipeline_entities.pipeline_execution.dataclasses.pipeline_component_instantiation_info import PipelineComponentInstantiationInfo
from data_classes.pipeline_configuration.pipeline_configuration import PipelineConfiguration


class MaxPredecessorsConstraint(StaticConstraint):
    """
    Static constraint that enforces a maximum number of predecessor nodes for a given node in the pipeline's DAG.
    """

    ##############################
    ### Attributs of instances ###
    ##############################
    _max_amount_: int
    _error_message_: str | None


    ###################
    ### Constructor ###
    ###################
    def __init__(self, max_amount: int):
        """
        Args:
            max_amount: The maximum allowed number of predecessor nodes.
        """

        self._max_amount_ = max_amount
        self._error_message_ = None


    ######################
    ### Public methods ###
    ######################
    def evaluate(self, own_node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo], pipeline_configuration: PipelineConfiguration) -> bool:
        """
        Evaluate whether the given node satisfies the maximum number of predecessors.

        Args:
            own_node (DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]):
                The node whose predecessors are checked.
            pipeline_configuration (PipelineConfiguration):
                The pipeline configuration.

        Returns:
            bool: True if the node has no more than the allowed number of predecessors, False otherwise.
        """

        if len(own_node.predecessors) <= self._max_amount_:
            self._error_message_ = None
            return True
        else:
            self._error_message_ = (
                f"Too many predecessors: found {len(own_node.predecessors)}, "
                f"but the maximum allowed number is {self._max_amount_}."
            )
            return False


    def get_error_message(self) -> str | None:
        """
        Retrieve the last error message generated during evaluation, if any.

        Returns:
            str | None: Error message if evaluation failed, otherwise None.
        """

        return self._error_message_


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
        if not isinstance(other, self.__class__):  # Covers None
            return False
        else:
            return self._max_amount_ == other._max_amount_
