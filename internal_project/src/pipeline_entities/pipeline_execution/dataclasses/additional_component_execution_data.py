from dataclasses import dataclass

from general_data_structures.directed_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode
from pipeline_entities.pipeline_execution.output.pipeline_component_execution_report import PipelineComponentExecutionReport
from pipeline_entities.pipeline_execution.dataclasses.pipeline_component_instantiation_info import \
    PipelineComponentInstantiationInfo
from pipeline_entities.large_data_classes.pipeline_configuration.pipeline_configuration import PipelineConfiguration
from pipeline_entities.large_data_classes.pipeline_input.pipeline_input import PipelineInput


@dataclass(frozen=True)
class AdditionalComponentExecutionData:
    overridden_attributes: dict[str, object]

    pipeline_configuration: PipelineConfiguration
    pipeline_input: PipelineInput

    own_graph_node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]

    component_execution_reports: dict[int, PipelineComponentExecutionReport]
