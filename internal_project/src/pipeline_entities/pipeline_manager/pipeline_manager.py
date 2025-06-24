import time
from copy import deepcopy
from dataclasses import fields

from data_structures.tree.tree import Tree
from data_structures.tree.tree_node import TreeNode
from exceptions.pipeline_execution_attribute_unmodified_exception import PipelineExecutionAttributeUnmodifiedException
from exceptions.pipeline_execution_constraint_exception import PipelineExecutionConstraintException
from pipeline_entities.component_info.dataclasses.pipeline_component_info import PipelineComponentInfo
from pipeline_entities.components.abstracts.input_pipeline_component import InputPipelineComponent
from pipeline_entities.components.abstracts.pipeline_component import PipelineComponent
from pipeline_entities.constraints.abstracts.dynamic_constraint import DynamicConstraint
from pipeline_entities.constraints.abstracts.mixed_constraint import MixedConstraint
from pipeline_entities.data_transfer.pipeline_data import PipelineData
from pipeline_entities.pipeline.pipeline import Pipeline
from pipeline_entities.pipeline_manager.pipeline_execution_statistics import PipelineExecutionStatistics


class PipelineManager:
    ###############################
    ### Attributes of instances ###
    ###############################
    _pipeline_: Pipeline

    _execution_stack_: list[TreeNode[PipelineComponentInfo]]
    _pipeline_data_dict_: dict[str, PipelineData]
    _pipeline_execution_statistics_: PipelineExecutionStatistics



    ###################
    ### Constructor ###
    ###################
    def __init__(self, pipeline: Pipeline):
        self._pipeline_ = pipeline

        component_tree: Tree[PipelineComponentInfo] = self._pipeline_.pipeline_configuration.components

        self._execution_stack_ = list(component_tree.pre_order_traversal())
        self._execution_stack_.reverse()

        self._pipeline_data_dict_ = {}
        self._pipeline_execution_statistics_ = PipelineExecutionStatistics()



    ######################
    ### Public methods ###
    ######################
    def execute_all(self) -> None:
        while not self.is_completely_executed:
            self.execute_next_component()



    def execute_next_component(self) -> None:
        node: TreeNode[PipelineComponentInfo] = self._execution_stack_.pop()
        pipeline_data: PipelineData = self._get_pipeline_data_for_node_(node)

        self._validate_node_component_constraints_(node, pipeline_data)

        pipeline_component: PipelineComponent = self._init_node_component_(node, pipeline_data)
        self._execute_node_component_(pipeline_component, node)

        self._validate_node_component_changes_(node, pipeline_data)



    #########################
    ### Getters & setters ###
    #########################
    @property
    def is_completely_executed(self) -> bool:
        return not self._execution_stack_



    #######################
    ### Private methods ###
    #######################
    def _validate_node_component_constraints_(self, node: TreeNode[PipelineComponentInfo], pipeline_data: PipelineData) -> None:
        self._validate_dynamic_node_component_constraints_(node, pipeline_data)
        self._validate_mixed_node_component_constraints_(node, pipeline_data)



    def _validate_dynamic_node_component_constraints_(self, node: TreeNode[PipelineComponentInfo], pipeline_data: PipelineData) -> None:
        dynamic_constraints: list[DynamicConstraint] = node.value.component_meta_info.dynamic_constraints

        for dynamic_constraint in dynamic_constraints:
            if not dynamic_constraint.evaluate(pipeline_data, self._pipeline_.pipeline_input):
                raise PipelineExecutionConstraintException(f"The dynamic constraint '{dynamic_constraint}' of the component"
                    f"'{node.value.component_id}' at the path '{node.path}' was violated.", node, dynamic_constraint)



    def _validate_mixed_node_component_constraints_(self, node: TreeNode[PipelineComponentInfo], pipeline_data: PipelineData) -> None:
        mixed_constraints: list[MixedConstraint] = node.value.component_meta_info.mixed_constraints
        for mixed_constraint in mixed_constraints:
            if not mixed_constraint.evaluate(pipeline_data, self._pipeline_.pipeline_input, node, self._pipeline_.pipeline_configuration):
                raise PipelineExecutionConstraintException(f"The mixed constraint '{mixed_constraint}' of the component"
                    f"'{node.value.component_id}' at the path '{node.path}' was violated.", node, mixed_constraint)



    def _init_node_component_(self, node: TreeNode[PipelineComponentInfo],
                              pipeline_data: PipelineData) -> PipelineComponent:
        component_cls: type = node.value.component_class

        if issubclass(component_cls, InputPipelineComponent):
            init_start: float = time.perf_counter()
            pipeline_component: PipelineComponent = component_cls(self._pipeline_.pipeline_input, pipeline_data)
            init_end: float = time.perf_counter()
        elif issubclass(component_cls, PipelineComponent):
            init_start: float = time.perf_counter()
            pipeline_component: PipelineComponent = component_cls(pipeline_data)
            init_end: float = time.perf_counter()
        else:
            raise ValueError(f"The PipelineComponentInfo for id '{node.value.component_id}' contains a component_class which is no subclass of PipelineComponent (type: '{component_cls}').")

        self._pipeline_execution_statistics_.component_init_durations[node.path] = init_end - init_start
        return pipeline_component



    def _execute_node_component_(self, pipeline_component: PipelineComponent, node: TreeNode[PipelineComponentInfo]) -> None:
        execution_start: float = time.perf_counter()

        pipeline_component.perform_action()

        execution_end: float = time.perf_counter()

        self._pipeline_execution_statistics_.component_execution_durations[node.path] = execution_end - execution_start



    def _validate_node_component_changes_(self, node: TreeNode[PipelineComponentInfo], pipeline_data: PipelineData) -> None:
        attributes_modifying: set[str] = node.value.component_meta_info.attributes_modifying

        for attribute_name in attributes_modifying:
            if any(field.name == attribute_name for field in fields(PipelineData)):
                if getattr(pipeline_data, attribute_name) is not None:
                    continue
            else:
                if attribute_name in pipeline_data.additional_values and pipeline_data.additional_values[attribute_name] is not None:
                    continue

            raise PipelineExecutionAttributeUnmodifiedException(f"The attribute '{attribute_name}' of the PipelineData"
                    f"was not modified through the component '{node.value.component_id}' at the path '{node.path}'"
                    f"even though the component should.", node, attribute_name)



    def _get_pipeline_data_for_node_(self, node: TreeNode[PipelineComponentInfo]) -> PipelineData:
        component_tree: Tree[PipelineComponentInfo] = self._pipeline_.pipeline_configuration.components
        root_node: TreeNode[PipelineComponentInfo] = component_tree.root_node

        if node is root_node:
            pipeline_data: PipelineData = PipelineData()
            self._pipeline_data_dict_[node.path] = pipeline_data
            return pipeline_data
        else:
            parent_pipeline_data: PipelineData = self._pipeline_data_dict_[node.parent_node.path]
            new_pipeline_data: PipelineData = deepcopy(parent_pipeline_data)
            self._pipeline_data_dict_[node.path] = new_pipeline_data
            return new_pipeline_data


