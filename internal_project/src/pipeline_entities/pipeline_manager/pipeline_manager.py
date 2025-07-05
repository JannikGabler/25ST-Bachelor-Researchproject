import time
import jax.numpy as jnp
from collections import deque
from copy import deepcopy
from dataclasses import fields

from data_structures.directed_acyclic_graph.directional_acyclic_graph import DirectionalAcyclicGraph
from data_structures.directed_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode
from exceptions.pipeline_constraint_violation_exception import PipelineConstraintViolationException
from exceptions.pipeline_execution_attribute_modified_exception import PipelineExecutionAttributeModifiedException
from exceptions.pipeline_execution_attribute_unmodified_exception import PipelineExecutionAttributeUnmodifiedException
from pipeline_entities.component_info.dataclasses.pipeline_component_info import PipelineComponentInfo
from pipeline_entities.components.abstracts.pipeline_component import PipelineComponent
from pipeline_entities.constraints.abstracts.dynamic_constraint import DynamicConstraint
from pipeline_entities.data_transfer.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.output.pipeline_component_execution_report import PipelineComponentExecutionReport
from pipeline_entities.data_transfer.pipeline_data import PipelineData
from pipeline_entities.pipeline.pipeline import Pipeline
from pipeline_entities.pipeline_component_instantiation_info.pipeline_component_instantiation_info import \
    PipelineComponentInstantiationInfo
from pipeline_entities.pipeline_configuration.dataclasses.pipeline_configuration import PipelineConfiguration
from pipeline_entities.pipeline_input.pipeline_input import PipelineInput


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
        self._pipeline_ = pipeline

        components_dag: DirectionalAcyclicGraph[PipelineComponentInstantiationInfo] = self._pipeline_.pipeline_configuration.components

        self._execution_stack_ = deque(components_dag.topological_traversal())
        self._component_execution_reports_ = {}



    ######################
    ### Public methods ###
    ######################
    def execute_all(self) -> None:
        while not self.is_completely_executed:
            self.execute_next_component()



    def execute_next_component(self) -> None:
        node_in_own_dag: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo] = self._execution_stack_.popleft()
        self._init_component_execution_record_(node_in_own_dag)
        pipeline_data, additional_info = self._get_input_for_node_(node_in_own_dag)

        PipelineManager._validate_node_component_constraints_(pipeline_data, additional_info)

        pipeline_component: PipelineComponent = self._init_node_component_(node_in_own_dag, pipeline_data, additional_info)
        result_pipeline_data: PipelineData = self._execute_node_component_(pipeline_component, node_in_own_dag)
        self._component_execution_reports_[id(node_in_own_dag)].component_output = result_pipeline_data

        self._validate_node_component_changes_(result_pipeline_data, node_in_own_dag)



    #########################
    ### Getters & setters ###
    #########################
    @property
    def is_completely_executed(self) -> bool:
        return not self._execution_stack_



    #######################
    ### Private methods ###
    #######################
    def _init_component_execution_record_(self, node_in_own_dag: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]) -> None:
        execution_report: PipelineComponentExecutionReport = PipelineComponentExecutionReport()
        execution_report.component_instantiation_info = node_in_own_dag.value

        self._component_execution_reports_[id(node_in_own_dag)] = execution_report



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



    def _execute_node_component_(self, pipeline_component: PipelineComponent,
                                 node_in_own_dag: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]) -> PipelineData:

        execution_start: float = time.perf_counter()

        result_pipeline_data: PipelineData = pipeline_component.perform_action()

        execution_end: float = time.perf_counter()

        self._component_execution_reports_[id(node_in_own_dag)].component_execution_time = execution_end - execution_start
        return result_pipeline_data



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


    
    def _ascii_dag(self, adj, labels):
        """
        Draw an ASCII representation of a DAG.
        - adj: dict[node, list of successor nodes]
        - labels: dict[node, str label]
        """
        # Build reverse adjacency for level assignment
        rev = {u: [] for u in adj}
        for u, vs in adj.items():
            for v in vs:
                rev.setdefault(v, []).append(u)

        # Compute levels (distance from sources)
        levels = {}
        def compute_level(u):
            if u in levels:
                return levels[u]
            if not rev.get(u):
                levels[u] = 0
            else:
                levels[u] = max(compute_level(p) for p in rev[u]) + 1
            return levels[u]

        for u in adj:
            compute_level(u)

        # Group nodes by level
        layers = {}
        for u, lvl in levels.items():
            layers.setdefault(lvl, []).append(u)
        for lvl in layers:
            layers[lvl].sort(key=lambda n: str(n))

        # Calculate box and spacing sizes
        max_label_len = max(len(labels[u]) for u in adj)
        box_w = max_label_len + 4  # padding
        box_h = 3
        h_spacing = 4
        v_spacing = 1

        # Assign coordinates
        coords = {}
        for lvl, nodes in layers.items():
            for idx, u in enumerate(nodes):
                x = lvl * (box_w + h_spacing)
                y = idx * (box_h + v_spacing)
                coords[u] = (x, y)

        # Canvas size
        width = max(x + box_w for x, y in coords.values()) + 1
        height = max(y + box_h for x, y in coords.values()) + 1

        # Initialize canvas
        canvas = [[' ' for _ in range(width)] for _ in range(height)]

        # Draw boxes
        for u, (x, y) in coords.items():
            label = labels[u]
            top = y
            mid = y + 1
            bot = y + 2
            # Top border
            for i in range(box_w):
                canvas[top][x + i] = '='
                canvas[bot][x + i] = '='
            # Middle content
            text = f"  {label}  ".center(box_w)
            canvas[mid][x] = canvas[mid][x + box_w - 1] = '='
            for i, ch in enumerate(text):
                canvas[mid][x + 1 + i] = ch

        # Draw edges
        arrow_margin = 1  # leave this many spaces between box and arrow
        for u, vs in adj.items():
            x0, y0 = coords[u]
            x0r = x0 + box_w
            y0m = y0 + 1
            for v in vs:
                x1, y1 = coords[v]
                x1l = x1
                y1m = y1 + 1

                # Horizontal line, inset by arrow_margin so we don't touch box borders
                for x in range(x0r + arrow_margin, x1l - arrow_margin - 1):
                    if canvas[y0m][x] == ' ':
                        canvas[y0m][x] = '-'

                # Arrow head, also inset
                arrow_x = x1l - arrow_margin - 1
                canvas[y0m][arrow_x] = '>'

                # Vertical connector if needed (at the same arrow_x)
                if y0m < y1m:
                    for yy in range(y0m + 1, y1m):
                        if canvas[yy][arrow_x] == ' ':
                            canvas[yy][arrow_x] = 'v'
                elif y0m > y1m:
                    for yy in range(y1m + 1, y0m):
                        if canvas[yy][arrow_x] == ' ':
                            canvas[yy][arrow_x] = '^'

        # Restore mid-line borders for closed boxes
        for u, (x, y) in coords.items():
            mid = y + 1
            canvas[mid][x] = '='
            canvas[mid][x + box_w - 1] = '='

        return "\n".join("".join(row) for row in canvas)
    

    def __repr__(self) -> str:
        dag   = self._pipeline_.pipeline_configuration.components
        nodes = list(dag.topological_traversal())

        adj = {}
        labels = {}
        for node in nodes:
            inst = node.value
            comp = inst.component

            comp_id = getattr(comp, "component_id", "error reading id!")
            comp_name = getattr(inst, "component_name", "error reading name!")

            adj[comp_id] = [
                getattr(succ.value.component, "component_id", "error reading id!")
                for succ in node.successors
            ]
            labels[comp_id] = f"{comp_name} {comp_id}"

        # 3) delegate to your ASCII‐drawing helper
        return self._ascii_dag(adj, labels)


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


