from __future__ import annotations

from general_data_structures.directional_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode
from pipeline_entities.pipeline_execution.dataclasses.pipeline_component_instantiation_info import PipelineComponentInstantiationInfo


class PipelineExecutionAttributeModifiedException(Exception):
    """
    Exception raised when a pipeline execution attribute is unexpectedly modified.
    """


    _causing_node_: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]
    _causing_attribute_name_: str


    def __init__(self, msg: str, causing_node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo], causing_attribute_name: str) -> None:
        """
        Args:
            msg (str): Description of the error.
            causing_node (DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]): The pipeline node in which the modification occurred.
            causing_attribute_name (str): The name of the attribute that was modified.
        """

        super().__init__(msg)
        self._causing_node_ = causing_node
        self._causing_attribute_name_ = causing_attribute_name


    @property
    def causing_pipeline_component(self) -> DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]:
        """
        Return the pipeline node in which the attribute modification occurred.

        Returns:
            DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]: The node that caused the modification.
        """

        return self._causing_node_


    @property
    def causing_attribute_name(self) -> str:
        """
        Return the name of the attribute that was modified.

        Returns:
            str: The modified attribute's name.
        """

        return self._causing_attribute_name_
