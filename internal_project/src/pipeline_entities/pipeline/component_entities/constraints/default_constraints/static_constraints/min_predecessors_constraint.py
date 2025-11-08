from general_data_structures.directional_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode

from pipeline_entities.pipeline.component_entities.constraints.abstracts.static_constraint import StaticConstraint
from pipeline_entities.pipeline_execution.dataclasses.pipeline_component_instantiation_info import PipelineComponentInstantiationInfo
from data_classes.pipeline_configuration.pipeline_configuration import PipelineConfiguration



class MinPredecessorsConstraint(StaticConstraint):
    """
    Static constraint that enforces a minimum number of predecessor nodes for a given node in the pipeline's DAG.
    """

    ##############################
    ### Attributs of instances ###
    ##############################
    _min_amount_: int
    _error_message_: str | None


    ###################
    ### Constructor ###
    ###################
    def __init__(self, min_amount: int):
        """
        Args:
            min_amount: The required minimum number of predecessor nodes.
        """

        self._min_amount_ = min_amount
        self._error_message_ = None


    ######################
    ### Public methods ###
    ######################
    def evaluate(self, own_node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo], pipeline_configuration: PipelineConfiguration) -> bool:
        """
        Evaluate whether the given node satisfies the minimum number of predecessors.

        Args:
            own_node (DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]):
                The node whose predecessors are checked.
            pipeline_configuration (PipelineConfiguration):
                The pipeline configuration.

        Returns:
            bool: True if the node has at least the required number of predecessors, False otherwise.
        """

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
        """
        Retrieve the error message generated during evaluation, if any.

        Returns:
            str | None: Error message if evaluation failed, otherwise None.
        """

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
        if not isinstance(other, self.__class__):  # Covers None
            return False
        else:
            return self._min_amount_ == other._min_amount_
