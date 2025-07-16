import time
import jax.numpy as jnp
from collections import deque
from copy import deepcopy
from dataclasses import fields

from general_data_structures.directed_acyclic_graph.directional_acyclic_graph import DirectionalAcyclicGraph
from general_data_structures.directed_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode
from exceptions.pipeline_constraint_violation_exception import PipelineConstraintViolationException
from exceptions.pipeline_execution_attribute_modified_exception import PipelineExecutionAttributeModifiedException
from exceptions.pipeline_execution_attribute_unmodified_exception import PipelineExecutionAttributeUnmodifiedException
from pipeline_entities.pipeline.component_entities.component_info.dataclasses.pipeline_component_info import PipelineComponentInfo
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component import PipelineComponent
from pipeline_entities.pipeline.component_entities.constraints.abstracts.dynamic_constraint import DynamicConstraint
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.pipeline_execution.output.pipeline_component_execution_report import PipelineComponentExecutionReport
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline.data_classes.pipeline import Pipeline
from pipeline_entities.pipeline_execution.dataclasses.pipeline_component_instantiation_info import \
    PipelineComponentInstantiationInfo
from pipeline_entities.large_data_classes.pipeline_configuration.pipeline_configuration import PipelineConfiguration
from pipeline_entities.large_data_classes.pipeline_input.pipeline_input import PipelineInput


class PipelineManager:
    ###############################
    ### Attributes of instances ###
    ###############################
    _pipeline_: Pipeline

    _execution_stack_: deque[DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]]
    _component_execution_reports_: dict[int, PipelineComponentExecutionReport]



    ###################
    ### Constructor ###
    ###################
    def __init__(self, pipeline: Pipeline):
        self._pipeline_ = deepcopy(pipeline)

        components_dag: DirectionalAcyclicGraph[PipelineComponentInstantiationInfo] = self._pipeline_.pipeline_configuration.components

        self._execution_stack_ = deque(components_dag.topological_traversal())
        self._component_execution_reports_ = {}



    ######################
    ### Public methods ###
    ######################
    def execute_all(self) -> None:
        while not self.is_completely_executed:
            self.execute_next_component()



    def execute_next_component(self) -> PipelineComponentExecutionReport:
        node_in_own_dag: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo] = self._execution_stack_.popleft()
        pipeline_data, additional_execution_data = self._setup_node_execution_(node_in_own_dag)
        pipeline_component: PipelineComponent = self._init_node_component_(node_in_own_dag, pipeline_data, additional_execution_data)
        result_data: PipelineData = self._execute_node_component_(pipeline_component, node_in_own_dag)
        self._finish_up_node_execution(node_in_own_dag, result_data)
        return deepcopy(self._component_execution_reports_[id(node_in_own_dag)])



    def get_component_execution_report(self, pipeline_component_name: str) -> PipelineComponentExecutionReport:
        pipeline_configuration: PipelineConfiguration = self._pipeline_.pipeline_configuration
        node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo] = pipeline_configuration.get_component_node_by_component_name(pipeline_component_name)

        return deepcopy(self._component_execution_reports_[id(node)])



    def get_all_component_execution_report(self) -> list[PipelineComponentExecutionReport]:
        return deepcopy(list(self._component_execution_reports_.values()))



    #########################
    ### Getters & setters ###
    #########################
    @property
    def is_completely_executed(self) -> bool:
        return not self._execution_stack_

    @property
    def pipeline(self) -> Pipeline:
        return deepcopy(self._pipeline_)



    ##########################
    ### Overridden methods ###
    ##########################
    def __eq__(self, other) -> bool:
        if not isinstance(other, PipelineManager):
            return False
        else:
            return self._pipeline_ == other._pipeline_



    def __str__(self) -> str:
        return (f"PipelineManager(pipeline_configuration_name={repr(self._pipeline_.pipeline_configuration.name)}, "
                f"pipeline_input_name={repr(self._pipeline_.pipeline_input.name)}, amount_of_nodes_left_to_execute={len(self._execution_stack_)})")



    def __repr__(self) -> str:
        return (f"PipelineManager(pipeline={repr(self._pipeline_)}, execution_stack={repr(self._execution_stack_)},"
                f"component_execution_reports={repr(self._component_execution_reports_)})")



    #######################
    ### Private methods ###
    #######################

    ### Node Execution ###

    def _execute_node_component_(self, pipeline_component: PipelineComponent,
                                 node_in_own_dag: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]) -> PipelineData:

        execution_start: float = time.perf_counter()

        result_pipeline_data: PipelineData = pipeline_component.perform_action()

        execution_end: float = time.perf_counter()

        self._component_execution_reports_[id(node_in_own_dag)].component_execution_time = execution_end - execution_start
        return result_pipeline_data



    ### Node execution setup ###

    def _setup_node_execution_(self, node_in_own_dag: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]) -> tuple[list[PipelineData], AdditionalComponentExecutionData]:
        self._init_component_execution_record_(node_in_own_dag)
        pipeline_data, additional_info = self._get_input_for_node_(node_in_own_dag)

        PipelineManager._validate_node_component_constraints_(pipeline_data, additional_info)

        return pipeline_data, additional_info


    def _init_component_execution_record_(self, node_in_own_dag: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]) -> None:
        execution_report: PipelineComponentExecutionReport = PipelineComponentExecutionReport()
        execution_report.component_instantiation_info = node_in_own_dag.value

        self._component_execution_reports_[id(node_in_own_dag)] = execution_report



    ### Node execution finish up ###

    def _finish_up_node_execution(self, node_in_own_dag: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                                  result_data: PipelineData) -> None:

        self._component_execution_reports_[id(node_in_own_dag)].component_output = result_data

        self._validate_node_component_changes_(result_data, node_in_own_dag)



    ### Input generation for node execution ###

    def _get_input_for_node_(self, node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]) -> tuple[list[PipelineData], AdditionalComponentExecutionData]:
        if len(node.predecessors) == 0:
            return self._create_new_input_for_entry_node_(node)
        else:
            return self._create_input_for_non_entry_node_(node)


    def _create_new_input_for_entry_node_(self, node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]) -> tuple[list[PipelineData], AdditionalComponentExecutionData]:
        index_of_old_node: int = self._pipeline_.pipeline_configuration.components.get_index_of_node_(node)

        new_configuration: PipelineConfiguration = deepcopy(self._pipeline_.pipeline_configuration)
        new_input: PipelineInput = deepcopy(self._pipeline_.pipeline_input)
        new_own_node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo] = new_configuration.components[index_of_old_node]

        additional_component_execution_data: AdditionalComponentExecutionData = (
            AdditionalComponentExecutionData(new_configuration, new_input, new_own_node, {}))

        new_pipeline_data: PipelineData = PipelineData()
        return [new_pipeline_data], additional_component_execution_data


    # TODO: split up (functional decomposition)
    def _create_input_for_non_entry_node_(self, node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]) -> tuple[list[PipelineData], AdditionalComponentExecutionData]:
        old_configuration: PipelineConfiguration = self._pipeline_.pipeline_configuration
        new_configuration: PipelineConfiguration = deepcopy(old_configuration)
        new_input: PipelineInput = deepcopy(self._pipeline_.pipeline_input)

        node_mapping: dict[int, DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]] = {}
        for old_node in old_configuration.components.breadth_first_traversal():
            old_node_index: int = old_configuration.components.get_index_of_node_(old_node)
            node_mapping[id(old_node)] = new_configuration.components[old_node_index]

        new_own_node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo] = node_mapping[id(node)]

        new_reports: dict[int, PipelineComponentExecutionReport] = {}
        for old_predecessor in node.all_predecessors_traversal():
            report: PipelineComponentExecutionReport = self._component_execution_reports_[id(old_predecessor)]
            new_predecessor: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo] = node_mapping[id(old_predecessor)]
            new_reports[id(new_predecessor)] = deepcopy(report)

        pipeline_data_list: list[PipelineData] = [new_reports[id(pre)].component_output for pre in new_own_node.predecessors]
        return pipeline_data_list, AdditionalComponentExecutionData(new_configuration, new_input, new_own_node, new_reports)



    ### Constraint validation after node execution ###

    @staticmethod
    def _validate_node_component_constraints_(pipeline_data: list[PipelineData], additional_execution_data: AdditionalComponentExecutionData) -> None:
        PipelineManager._validate_dynamic_node_component_constraints_(pipeline_data, additional_execution_data)
        #self._validate_mixed_node_component_constraints_(node, pipeline_data, additional_execution_info)


    @staticmethod
    def _validate_dynamic_node_component_constraints_(pipeline_data: list[PipelineData], additional_execution_data: AdditionalComponentExecutionData) -> None:
        node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo] = additional_execution_data.own_graph_node
        dynamic_constraints: list[DynamicConstraint] = node.value.component.component_meta_info.dynamic_constraints

        for dynamic_constraint in dynamic_constraints:
            if not dynamic_constraint.evaluate(pipeline_data, additional_execution_data):
                raise PipelineConstraintViolationException(f"The dynamic constraint {repr(dynamic_constraint)} of the component "
                    f"{repr(node.value.component_name)} with id {repr(node.value.component.component_id)} was violated.", node, dynamic_constraint)


    # TODO: delete
    # def _validate_mixed_node_component_constraints_(self, node: DirectionalAcyclicGraphNode[PipelineComponentInfo], pipeline_data: list[PipelineData],
    #                                                   additional_execution_info: AdditionalComponentExecutionData) -> None:
    #
    #     mixed_constraints: list[MixedConstraint] = node.value.component_meta_info.mixed_constraints
    #
    #     for mixed_constraint in mixed_constraints:
    #         if not mixed_constraint.evaluate(pipeline_data, self._pipeline_.pipeline_input, node, self._pipeline_.pipeline_configuration):
    #             raise PipelineConstraintViolationException(f"The mixed constraint '{mixed_constraint}' of the component"
    #                 f"'{node.value.component_id}' at the path '{node.path}' was violated.", node, mixed_constraint)


    def _init_node_component_(self, node_in_own_dag: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                              pipeline_data: list[PipelineData], additional_execution_data: AdditionalComponentExecutionData) -> PipelineComponent:

        execution_report: PipelineComponentExecutionReport = self._component_execution_reports_[id(node_in_own_dag)]
        component_info: PipelineComponentInfo = node_in_own_dag.value.component
        component_cls: type = component_info.component_class

        if issubclass(component_cls, PipelineComponent):
            init_start: float = time.perf_counter()
            pipeline_component: PipelineComponent = component_cls(pipeline_data, additional_execution_data)
            init_end: float = time.perf_counter()
        else:
            raise ValueError(
                f"The PipelineComponentInfo for id '{component_info.component_id}' contains a component_class which is no subclass of PipelineComponent (type: '{component_cls}').")

        execution_report.component_init_time = init_end - init_start
        return pipeline_component


    def _validate_node_component_changes_(self, result_pipeline_data: PipelineData,
                                          node_in_own_dag: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]) -> None:

        attributes_modifying: set[str] = node_in_own_dag.value.component.component_meta_info.attributes_modifying
        attribute_names: list[str] = [field.name for field in fields(PipelineData) if field.name != "additional_values"] + list(result_pipeline_data.additional_values.keys())

        for attribute_name in attribute_names:
            if attribute_name in attributes_modifying:
                PipelineManager._validate_that_component_did_modify_attribute_(attribute_name, result_pipeline_data, node_in_own_dag)
            else:
                self._validate_that_component_did_not_modify_attribute_(attribute_name, result_pipeline_data, node_in_own_dag)


    @staticmethod
    def _validate_that_component_did_modify_attribute_(attribute_name: str, result_pipeline_data: PipelineData,
                                                     node_in_own_dag: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]) -> None:

        direct_attribute_set: bool = attribute_name != "additional_values" and hasattr(result_pipeline_data, attribute_name) \
                                        and getattr(result_pipeline_data, attribute_name) is not None
        additional_value_set: bool = attribute_name in result_pipeline_data.additional_values \
                                        and result_pipeline_data.additional_values[attribute_name] is not None

        if not direct_attribute_set and not additional_value_set:
            node_value: PipelineComponentInstantiationInfo = node_in_own_dag.value

            raise PipelineExecutionAttributeUnmodifiedException(
                    f"The component {repr(node_value.component_name)} with the id {repr(node_value.component.component_id)} "
                    f"did not modify the attribute {repr(attribute_name)} of the result PipelineData even though the component"
                    "should.", node_in_own_dag, attribute_name)


    def _validate_that_component_did_not_modify_attribute_(self, attribute_name: str, result_pipeline_data: PipelineData,
                                                           node_in_own_dag: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]) -> None:
        if node_in_own_dag.value.component.component_meta_info.allow_additional_value_modifications_outside_specification:
            return

        if node_in_own_dag.predecessors:
            for predecessor in node_in_own_dag.predecessors:
                report: PipelineComponentExecutionReport = self._component_execution_reports_[id(predecessor)]
                output_of_predecessor: PipelineData = report.component_output

                if PipelineManager._is_attribute_equal_in_both_pipeline_datas_(attribute_name, output_of_predecessor, result_pipeline_data):
                    return
        else:
            input_data, additional_execution_data = self._create_new_input_for_entry_node_(node_in_own_dag)

            if PipelineManager._is_attribute_equal_in_both_pipeline_datas_(attribute_name, input_data[0], result_pipeline_data):
                return

        node_value: PipelineComponentInstantiationInfo = node_in_own_dag.value
        raise PipelineExecutionAttributeModifiedException(
            f"The component {repr(node_value.component_name)} with the id {repr(node_value.component.component_id)} did "
            f"modify the attribute {repr(attribute_name)} of the result PipelineData even though the component shouldn't "
            f"have.", node_in_own_dag, attribute_name)


    @staticmethod
    def _is_attribute_equal_in_both_pipeline_datas_(attribute_name: str, output_of_predecessor: PipelineData, result_pipeline_data: PipelineData) -> bool:
        if attribute_name in [field.name for field in fields(PipelineData)]:
            if isinstance(getattr(output_of_predecessor, attribute_name), jnp.ndarray):
                if jnp.array_equal(getattr(output_of_predecessor, attribute_name),
                                   getattr(result_pipeline_data, attribute_name)):
                    return True
            elif getattr(output_of_predecessor, attribute_name) == getattr(result_pipeline_data, attribute_name):
                return True
        else:
            if attribute_name in output_of_predecessor.additional_values \
                    and attribute_name in result_pipeline_data.additional_values \
                    and output_of_predecessor.additional_values[attribute_name] == \
                    result_pipeline_data.additional_values[attribute_name]:
                return True

        return False

    # def _get_pipeline_data_for_node_(self, node: TreeNode[PipelineComponentInfo]) -> PipelineData:
    #     component_tree: Tree[PipelineComponentInfo] = self._pipeline_.pipeline_configuration.components
    #     root_node: TreeNode[PipelineComponentInfo] = component_tree.root_node
    #
    #     if node is root_node:
    #         pipeline_data: PipelineData = PipelineData()
    #         self._pipeline_data_dict_[node.path] = pipeline_data
    #         return pipeline_data
    #     else:
    #         parent_pipeline_data: PipelineData = self._pipeline_data_dict_[node.parent_node.path]
    #         new_pipeline_data: PipelineData = deepcopy(parent_pipeline_data)
    #         self._pipeline_data_dict_[node.path] = new_pipeline_data
    #         return new_pipeline_data


