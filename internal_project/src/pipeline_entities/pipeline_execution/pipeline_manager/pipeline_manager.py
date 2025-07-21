import time
from collections import deque
from copy import deepcopy

from general_data_structures.directed_acyclic_graph.directional_acyclic_graph import DirectionalAcyclicGraph
from general_data_structures.directed_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode
from pipeline_entities.pipeline.component_entities.component_info.dataclasses.pipeline_component_info import PipelineComponentInfo
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component import PipelineComponent
from pipeline_entities.pipeline.pipeline import Pipeline
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.pipeline_execution.output.pipeline_component_execution_report import PipelineComponentExecutionReport
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline_execution.dataclasses.pipeline_component_instantiation_info import \
    PipelineComponentInstantiationInfo
from pipeline_entities.large_data_classes.pipeline_configuration.pipeline_configuration import PipelineConfiguration
from utils.pipeline_component_execution_validation_utils import PipelineComponentExecutionValidationUtils
from utils.pipeline_manager_utils import PipelineManagerUtils


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
        pipeline_data, additional_execution_data, old_attributes = self._setup_node_execution_(node_in_own_dag)
        pipeline_component: PipelineComponent = self._init_node_component_(node_in_own_dag, pipeline_data, additional_execution_data)
        result_data: PipelineData = self._execute_node_component_(pipeline_component, node_in_own_dag)
        self._finish_up_node_execution(node_in_own_dag, result_data, old_attributes)
        return deepcopy(self._component_execution_reports_[id(node_in_own_dag)])



    def get_component_execution_report(self, pipeline_component_name: str) -> PipelineComponentExecutionReport:
        pipeline_configuration: PipelineConfiguration = self._pipeline_.pipeline_configuration
        node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo] = pipeline_configuration.get_component_node_by_component_name(pipeline_component_name)

        return deepcopy(self._component_execution_reports_[id(node)])



    def get_all_component_execution_reports(self) -> list[PipelineComponentExecutionReport]:
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
    def __str__(self) -> str:
        return (f"PipelineManager(pipeline_configuration_name={repr(self._pipeline_.pipeline_configuration.name)}, "
                f"pipeline_input_name={repr(self._pipeline_.pipeline_input.name)}, amount_of_nodes_left_to_execute={len(self._execution_stack_)})")



    def __repr__(self) -> str:
        return (f"PipelineManager(pipeline={repr(self._pipeline_)}, execution_stack={repr(self._execution_stack_)},"
                f"component_execution_reports={repr(self._component_execution_reports_)})")



    def __hash__(self) -> int | None:
        return None



    def __eq__(self, other) -> bool:
        if not isinstance(other, PipelineManager):
            return False
        else:
            return self._pipeline_ == other._pipeline_



    #######################
    ### Private methods ###
    #######################

    ### Node initialisation ###
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
            raise ValueError(f"The PipelineComponentInfo for id '{component_info.component_id}' contains a "
                             f"component_class which is no subclass of PipelineComponent (type: '{component_cls}').")

        execution_report.component_init_time = init_end - init_start
        return pipeline_component



    ### Node Execution ###

    def _execute_node_component_(self, pipeline_component: PipelineComponent,
                                 node_in_own_dag: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]) -> PipelineData:

        execution_start: float = time.perf_counter()

        result_pipeline_data: PipelineData = pipeline_component.perform_action()

        execution_end: float = time.perf_counter()

        self._component_execution_reports_[id(node_in_own_dag)].component_execution_time = execution_end - execution_start
        return result_pipeline_data



    ### Node execution setup ###

    def _setup_node_execution_(self, node_in_own_dag: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]) \
                    -> tuple[list[PipelineData], AdditionalComponentExecutionData, dict[str, object]]:
        self._init_component_execution_record_(node_in_own_dag)
        pipeline_data, additional_info, old_attributes = PipelineManagerUtils.create_input_for_node(
            node_in_own_dag, self._pipeline_, self._component_execution_reports_)

        PipelineComponentExecutionValidationUtils.validate_before_node_execution(node_in_own_dag, self._pipeline_, self._component_execution_reports_)

        return pipeline_data, additional_info, old_attributes


    def _init_component_execution_record_(self, node_in_own_dag: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]) -> None:
        execution_report: PipelineComponentExecutionReport = PipelineComponentExecutionReport()
        execution_report.component_instantiation_info = node_in_own_dag.value

        self._component_execution_reports_[id(node_in_own_dag)] = execution_report



    ### Node execution finish up ###

    def _finish_up_node_execution(self, node_in_own_dag: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                                  result_data: PipelineData, old_attributes: dict[str, object]) -> None:

        PipelineManagerUtils.reverse_attribute_overrides(result_data, old_attributes)

        self._component_execution_reports_[id(node_in_own_dag)].component_output = result_data

        PipelineComponentExecutionValidationUtils.validate_after_node_execution(node_in_own_dag, self._pipeline_, self._component_execution_reports_)




