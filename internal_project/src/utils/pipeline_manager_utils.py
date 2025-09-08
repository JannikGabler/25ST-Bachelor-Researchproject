from copy import deepcopy
from dataclasses import fields

from exceptions.not_instantiable_error import NotInstantiableError
from general_data_structures.directional_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode
from pipeline_entities.large_data_classes.pipeline_configuration.pipeline_configuration import PipelineConfiguration
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.large_data_classes.pipeline_input.pipeline_input import PipelineInput
from pipeline_entities.pipeline.pipeline import Pipeline
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import \
    AdditionalComponentExecutionData
from pipeline_entities.pipeline_execution.dataclasses.pipeline_component_instantiation_info import \
    PipelineComponentInstantiationInfo
from pipeline_entities.pipeline_execution.output.pipeline_component_execution_report import \
    PipelineComponentExecutionReport


class PipelineManagerUtils:
    REMOVE_VALUE_MARKER = object()



    ###################
    ### Constructor ###
    ###################
    def __init__(self) -> None:
        raise NotInstantiableError(f"The class {repr(self.__class__.__name__)} cannot be instantiated.")



    ######################
    ### Public methods ###
    ######################
    @classmethod
    def create_input_for_node(cls, node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                    pipeline: Pipeline, reports: dict[int, PipelineComponentExecutionReport]) \
                    -> tuple[list[PipelineData], AdditionalComponentExecutionData, dict[str, object]]:

        data: list[PipelineData]
        additional_data: AdditionalComponentExecutionData

        if len(node.predecessors) == 0:
            data, additional_data = cls._create_new_input_for_entry_node_(node, pipeline)
        else:
            data, additional_data = cls._create_input_for_non_entry_node_(node, pipeline, reports)

        old_attributes: dict[str, object] = cls._insert_overridden_attributes_(node, data, additional_data)
        return data, additional_data, old_attributes



    @classmethod
    def reverse_attribute_overrides(cls, node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                                    output_data: PipelineData, old_attributes: dict[str, object]) -> None:
        for attribute_name, old_value in old_attributes.items():
            if attribute_name not in node.value.component.component_meta_info.attributes_modifying:
                cls._override_attribute_(attribute_name, old_value, [output_data])



    #######################
    ### Private methods ###
    #######################
    @classmethod
    def _create_new_input_for_entry_node_(cls, node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                    pipeline: Pipeline) -> tuple[list[PipelineData], AdditionalComponentExecutionData]:

        index_of_old_node: int = pipeline.pipeline_configuration.components.get_index_of_node_(node)

        new_configuration: PipelineConfiguration = deepcopy(pipeline.pipeline_configuration)
        new_input: PipelineInput = deepcopy(pipeline.pipeline_input)
        new_own_node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo] = new_configuration.components[index_of_old_node]

        additional_component_execution_data: AdditionalComponentExecutionData = (
            AdditionalComponentExecutionData({}, new_configuration, new_input, new_own_node, {}) # TODO: overridden attributes
        )

        new_pipeline_data: PipelineData = PipelineData()
        return [new_pipeline_data], additional_component_execution_data



    # TODO: split up (functional decomposition)
    @classmethod
    def _create_input_for_non_entry_node_(cls, node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                    pipeline: Pipeline, reports: dict[int, PipelineComponentExecutionReport]) \
                    -> tuple[list[PipelineData], AdditionalComponentExecutionData]:

        old_configuration: PipelineConfiguration = pipeline.pipeline_configuration
        new_configuration: PipelineConfiguration = deepcopy(old_configuration)
        new_input: PipelineInput = deepcopy(pipeline.pipeline_input)

        node_mapping: dict[int, DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]] = {}
        for old_node in old_configuration.components.breadth_first_traversal():
            old_node_index: int = old_configuration.components.get_index_of_node_(old_node)
            node_mapping[id(old_node)] = new_configuration.components[old_node_index]

        new_own_node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo] = node_mapping[id(node)]

        new_reports: dict[int, PipelineComponentExecutionReport] = {}
        for old_predecessor in node.all_predecessors_traversal():
            report: PipelineComponentExecutionReport = reports[id(old_predecessor)]
            new_predecessor: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo] = node_mapping[id(old_predecessor)]
            new_reports[id(new_predecessor)] = deepcopy(report)

        pipeline_data_list: list[PipelineData] = [new_reports[id(pre)].component_output for pre in new_own_node.predecessors]
        return pipeline_data_list, AdditionalComponentExecutionData({}, new_configuration, new_input, new_own_node, new_reports) # TODO: overridden attributes



    @classmethod
    def _insert_overridden_attributes_(cls, node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                    data: list[PipelineData], additional_data: AdditionalComponentExecutionData) -> dict[str, object]:

        overridden_attributes: dict[str, object] = deepcopy(node.value.overridden_attributes)
        old_attributes: dict[str, object] = {}

        additional_data.overridden_attributes.update(overridden_attributes)

        for attribute_name, attribute_value in overridden_attributes.items():
            old_value: object = cls._override_attribute_(attribute_name, attribute_value, data)
            old_attributes[attribute_name] = old_value

        return old_attributes



    @classmethod
    def _override_attribute_(cls, attribute_name: str, attribute_value: object, data: list[PipelineData]) -> object:
        old_value: object

        if attribute_name != "additional_values" and any(field.name == attribute_name for field in fields(PipelineData)):
            old_value = getattr(data[0], attribute_name)

            for pipeline_data in data:
                setattr(pipeline_data, attribute_name, attribute_value)

        else:
            old_value = data[0].additional_values.get(attribute_name, cls.REMOVE_VALUE_MARKER)

            for pipeline_data in data:
                if attribute_value is cls.REMOVE_VALUE_MARKER:
                    del pipeline_data.additional_values[attribute_name]
                else:
                    pipeline_data.additional_values[attribute_name] = attribute_value

        return old_value