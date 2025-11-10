from dataclasses import dataclass

from data_classes.pipeline_configuration.pipeline_configuration import PipelineConfiguration
from data_classes.pipeline_input.pipeline_input import PipelineInput
from utils.directional_acyclic_graph_utils import DirectionalAcyclicGraphUtils


@dataclass(frozen=True)
class Pipeline:
    """
    Immutable container for a pipeline configuration and its input.
    Provides an ASCII visualization of the component graph.
    """

    pipeline_configuration: PipelineConfiguration
    pipeline_input: PipelineInput


    def __repr__(self) -> str:
        """
        ASCII representation of the pipeline's component DAG.

        Returns:
            str: Formatted DAG with component names and IDs.
        """

        dag   = self.pipeline_configuration.components
        nodes = list(dag.topological_traversal())

        adj = {}
        labels = {}
        for node in nodes:
            inst = node.value
            comp = inst.component

            comp_id = getattr(comp, "component_id", "error reading id!")
            comp_name = getattr(inst, "component_name", "error reading name!")

            adj[comp_id] = [getattr(succ.value.component, "component_id", "error reading id!") for succ in node.successors]
            labels[comp_id] = f"{comp_name} {comp_id}"

        return DirectionalAcyclicGraphUtils.ascii_dag(adj, labels)
