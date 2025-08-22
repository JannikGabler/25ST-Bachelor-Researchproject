from copy import deepcopy

import jax.numpy as jnp

from dataclasses import fields

from exceptions.not_instantiable_error import NotInstantiableError
from exceptions.pipeline_constraint_violation_exception import PipelineConstraintViolationException
from exceptions.pipeline_execution_attribute_modified_exception import PipelineExecutionAttributeModifiedException
from exceptions.pipeline_execution_attribute_unmodified_exception import PipelineExecutionAttributeUnmodifiedException
from general_data_structures.directional_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline.component_entities.constraints.abstracts.post_dynamic_constraint import \
    PostDynamicConstraint
from pipeline_entities.pipeline.component_entities.constraints.abstracts.pre_dynamic_constraint import \
    PreDynamicConstraint
from pipeline_entities.pipeline.pipeline import Pipeline
from pipeline_entities.pipeline_execution.dataclasses.pipeline_component_instantiation_info import \
    PipelineComponentInstantiationInfo
from pipeline_entities.pipeline_execution.output.pipeline_component_execution_report import \
    PipelineComponentExecutionReport
from utils.pipeline_manager_utils import PipelineManagerUtils


class PipelineComponentExecutionValidationUtils:
    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        raise NotInstantiableError(f"The class {repr(self.__class__.__name__)} can not be instantiated.")



    ######################
    ### Public methods ###
    ######################
    @classmethod
    def validate_before_node_execution(cls, node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                                       pipeline: Pipeline, reports: dict[int, PipelineComponentExecutionReport]) -> None:

        cls._validate_dynamic_node_component_pre_constraints_(node, pipeline, reports)



    @classmethod
    def validate_after_node_execution(cls, node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                                      pipeline: Pipeline, reports: dict[int, PipelineComponentExecutionReport]) -> None:

        cls._validate_node_component_changes_(node, reports)

        cls._validate_dynamic_node_component_post_constraints_(node, pipeline, reports)



    #######################
    ### Private methods ###
    #######################

    ### Constraint validation before node execution ###

    @classmethod
    def _validate_dynamic_node_component_pre_constraints_(cls, node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                      pipeline: Pipeline, reports: dict[int, PipelineComponentExecutionReport]) -> None:

        pre_dynamic_constraints: list[PreDynamicConstraint] = node.value.component.component_meta_info.pre_dynamic_constraints
        input_data = PipelineManagerUtils.create_input_for_node(node, pipeline, reports)

        for constraint in pre_dynamic_constraints:
            if not constraint.evaluate(input_data[0], input_data[1]):
                raise PipelineConstraintViolationException(
                    f"The pre dynamic constraint {repr(constraint)} of the component "
                    f"{repr(node.value.component_name)} with id {repr(node.value.component.component_id)} was violated. "
                    f"The constraint provided the following reason:\n\n"
                    f"{repr(constraint.get_error_message())}", node, constraint)



    ### Constraint validation after node execution ###

    @classmethod
    def _validate_dynamic_node_component_post_constraints_(cls, node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                    pipeline: Pipeline, reports: dict[int, PipelineComponentExecutionReport]) -> None:

        post_dynamic_constraints: list[PostDynamicConstraint] = node.value.component.component_meta_info.post_dynamic_constraints
        input_data = PipelineManagerUtils.create_input_for_node(node, pipeline, reports)
        output_data = deepcopy(reports[id(node)].component_output)

        for constraint in post_dynamic_constraints:
            if not constraint.evaluate(input_data, output_data):
                raise PipelineConstraintViolationException(
                    f"The post dynamic constraint {repr(constraint)} of the component "
                    f"{repr(node.value.component_name)} with id {repr(node.value.component.component_id)} was violated. "
                    f"The constraint provided the following reason:\n\n"
                    f"{repr(constraint.get_error_message())}", node, constraint)



    @classmethod
    def _validate_node_component_changes_(cls, node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                                          reports: dict[int, PipelineComponentExecutionReport]) -> None:
        result_data: PipelineData = reports[id(node)].component_output

        attributes_modifying: set[str] = node.value.component.component_meta_info.attributes_modifying
        attribute_names: list[str] = ([field.name for field in fields(PipelineData) if field.name != "additional_values"]
                                      + list(result_data.additional_values.keys()))

        for attribute_name in attribute_names:
            if attribute_name in attributes_modifying:
                cls._validate_that_component_did_modify_attribute_(attribute_name, node, reports)
            else:
                cls._validate_that_component_did_not_modify_attribute_(attribute_name, node, reports)



    @classmethod
    def _validate_that_component_did_modify_attribute_(cls, attribute_name: str, node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                    reports: dict[int, PipelineComponentExecutionReport]) -> None:

        result_data: PipelineData = reports[id(node)].component_output

        direct_attribute_set: bool = (attribute_name != "additional_values" and hasattr(result_data, attribute_name)
                                      and getattr(result_data, attribute_name) is not None)

        additional_value_set: bool = (attribute_name in result_data.additional_values
                                     and result_data.additional_values[attribute_name] is not None)

        if not direct_attribute_set and not additional_value_set:
            node_value: PipelineComponentInstantiationInfo = node.value

            raise PipelineExecutionAttributeUnmodifiedException(
                f"The component {repr(node_value.component_name)} with the id {repr(node_value.component.component_id)} "
                f"did not modify the attribute {repr(attribute_name)} of the result PipelineData even though the component"
                "should.", node, attribute_name)



    @classmethod
    def _validate_that_component_did_not_modify_attribute_(cls, attribute_name: str, node: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo],
                                                           reports: dict[int, PipelineComponentExecutionReport]) -> None:
        result_data: PipelineData = reports[id(node)].component_output

        if node.value.component.component_meta_info.allow_additional_value_modifications_outside_specification:
            return

        if node.predecessors:
            for predecessor in node.predecessors:
                report: PipelineComponentExecutionReport = reports[id(predecessor)]
                output_of_predecessor: PipelineData = report.component_output

                if cls._is_attribute_equal_in_both_pipeline_datas_(attribute_name, output_of_predecessor, result_data):
                    return
        else:
            input_data = PipelineData()

            if cls._is_attribute_equal_in_both_pipeline_datas_(attribute_name, input_data, result_data):
                return

        node_value: PipelineComponentInstantiationInfo = node.value
        raise PipelineExecutionAttributeModifiedException(
            f"The component {repr(node_value.component_name)} with the id {repr(node_value.component.component_id)} did "
            f"modify the attribute {repr(attribute_name)} of the result PipelineData even though the component shouldn't "
            f"have.", node, attribute_name)



    @classmethod
    def _is_attribute_equal_in_both_pipeline_datas_(cls, attribute_name: str, output_of_predecessor: PipelineData, result_pipeline_data: PipelineData) -> bool:
        if attribute_name in [field.name for field in fields(PipelineData)]:
            if isinstance(getattr(output_of_predecessor, attribute_name), jnp.ndarray):
                if jnp.array_equal(getattr(output_of_predecessor, attribute_name), getattr(result_pipeline_data, attribute_name)):
                    return True
            elif getattr(output_of_predecessor, attribute_name) == getattr(result_pipeline_data, attribute_name):
                return True
        else:
            # pred_value: object = output_of_predecessor.additional_values.get(attribute_name)
            # curr_value: object = result_pipeline_data.additional_values.get(attribute_name)
            #
            # if pred_value == curr_value:
            #     return True

            if (attribute_name in output_of_predecessor.additional_values
                and attribute_name in result_pipeline_data.additional_values
                and output_of_predecessor.additional_values[attribute_name] == result_pipeline_data.additional_values[attribute_name]):

                return True

        return False