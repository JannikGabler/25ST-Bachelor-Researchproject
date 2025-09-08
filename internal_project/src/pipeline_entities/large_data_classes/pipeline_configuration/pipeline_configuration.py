from __future__ import annotations
from typing import TYPE_CHECKING

import math
import types
import typing
from dataclasses import fields
from typing import Any

import jax
import jax.numpy as jnp
import numpy

from constants.internal_logic_constants import PipelineConfigurationConstants
from exceptions.prohibited_attribute_override_exception import ProhibitedAttributeOverrideException
from general_data_structures.directional_acyclic_graph.directional_acyclic_graph import DirectionalAcyclicGraph
from general_data_structures.directional_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode
from general_data_structures.tree.tree_node import TreeNode
from exceptions.pipeline_constraint_violation_exception import PipelineConstraintViolationException
from exceptions.type_annotation_error import TypeAnnotationError
from file_handling.dynamic_module_loading.dynamic_module_loader import DynamicModuleLoader

from typing import Callable

from packaging.version import Version

from general_data_structures.tree.tree import Tree
from exceptions.no_such_pipeline_component_error import NoSuchPipelineComponentError
from pipeline_entities.large_data_classes.pipeline_configuration.pipeline_configuration_data import PipelineConfigurationData
from pipeline_entities.pipeline_execution.dataclasses.pipeline_component_instantiation_info import \
    PipelineComponentInstantiationInfo
from utils.python_eval_utils import PythonEvalUtils
from utils.typing_utils import TypingUtils

if TYPE_CHECKING:
    from pipeline_entities.pipeline.component_entities.component_meta_info.dataclasses.component_meta_info import \
        ComponentMetaInfo
    from pipeline_entities.pipeline.component_entities.component_info.dataclasses.pipeline_component_info import \
        PipelineComponentInfo


class PipelineConfiguration:

    ###############################
    ### Attributes of instances ###
    ###############################
    _name_: str | None
    _supported_program_version_: Version

    _components_: DirectionalAcyclicGraph[PipelineComponentInstantiationInfo]

    _runs_for_component_execution_time_measurements_: int

    _additional_values_: dict[str, Any]



    ###################
    ### Constructor ###
    ###################
    def __init__(self, input_data: PipelineConfigurationData) -> None:
        self._parse_input_data_(input_data)
        self._validate_attribute_types_()

        self._components_.freeze()

        self._check_for_static_constraint_violations_()
        self._check_for_prohibited_override_()



    # def __init__(self, pipeline_configuration_data: PipelineConfigurationData) -> None:
    #     self._pipeline_name_ = pipeline_configuration_data.name
    #     self._supported_program_version_ = pipeline_configuration_data.supported_program_version
    #     self._components_ = self.__get_components_from_ids__(pipeline_configuration_data.components)
    #
    #     self._components_.freeze()
    #
    #     irregularity = self.__determine_irregularity_in_pipeline_configuration_data__()
    #     if irregularity:
    #         raise InvalidConfigurationDataException(irregularity)



    ######################
    ### Public methods ###
    ######################
    def get_all_component_names(self) -> list[str]:
        return [node.value.component_name for node in self._components_]



    def get_component_node_by_component_name(self, component_name: str) -> DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo]:
        for node in self._components_:
            if node.value.component_name == component_name:
                return node

        raise NoSuchPipelineComponentError(f"There is no pipeline component in this pipeline configuration with the name {repr(component_name)}.")



    #########################
    ### Getters & setters ###
    #########################
    @property
    def name(self) -> str:
        return self._name_

    @property
    def supported_program_version(self) -> Version:
        return self._supported_program_version_

    @property
    def components(self) -> DirectionalAcyclicGraph[PipelineComponentInstantiationInfo]:
        return self._components_   #__components__ is frozen (immutable)

    @property
    def runs_for_component_execution_time_measurements(self) -> int:
        return self._runs_for_component_execution_time_measurements_

    @property
    def additional_values(self) -> dict[str, Any]:
        return self._additional_values_



    ##########################
    ### Overridden methods ###
    ##########################
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name={repr(self._name_)}, "
                f"supported_program_version={repr(self._supported_program_version_)}, "
                f"components={repr(self.components)}, "
                f"runs_for_component_execution_time_measurements={repr(self.runs_for_component_execution_time_measurements)}, "
                f"additional_values={repr(self.additional_values)})")

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}(name={str(self._name_)}, "
                f"supported_program_version={str(self._supported_program_version_)}, "
                f"components={str(self.components)}, "
                f"runs_for_component_execution_time_measurements={str(self.runs_for_component_execution_time_measurements)}, "
                f"additional_values={str(self.additional_values)})")



    def __eq__(self, other):
        if not isinstance(other, PipelineConfiguration):   # Covers None
            return False
        else:
            return (self._name_ == other._name_
                    and self._supported_program_version_ == other._supported_program_version_
                    and self._runs_for_component_execution_time_measurements_ == other._runs_for_component_execution_time_measurements_
                    and self._components_ == other._components_)



    def __hash__(self) -> int:   # Instances is immutable
        return hash((self._name_, self._supported_program_version_,
                     self._runs_for_component_execution_time_measurements_, self._components_))



    #######################
    ### Private methods ###
    #######################
    def _parse_input_data_(self, input_data: PipelineConfigurationData) -> None:
        eval_name_space: dict[str, Any] = DynamicModuleLoader.get_module_namespace() | PipelineConfigurationConstants.PARSING_EVAL_NAMESPACE

        self._parse_regular_input_values_(input_data, eval_name_space)
        self._parse_additional_input_values_(input_data, eval_name_space)
        self._parse_components_value_(input_data, eval_name_space)



    def _parse_regular_input_values_(self, input_data: PipelineConfigurationData, eval_name_space: dict[str, Any]) -> None:
        for field in fields(PipelineConfigurationData):
            name: str = field.name
            value: str = getattr(input_data, name)

            if name not in ("components", "additional_values"):
                self._parse_single_regular_input_value_(name, value, eval_name_space)



    def _parse_single_regular_input_value_(self, field_name: str, field_value: str, eval_name_space: dict[str, Any]) -> None:
        transformed_field_name: str = f"_{field_name}_"

        if field_value:
            parsed_value: Any = PythonEvalUtils.try_expression_evaluation(field_value, eval_name_space)
            setattr(self, transformed_field_name, parsed_value)
        else:
            setattr(self, transformed_field_name, None)



    def _parse_additional_input_values_(self, input_data: PipelineConfigurationData, eval_name_space: dict[str, Any]) -> None:
        self._additional_values_ = {}

        for key, value in input_data.additional_values.items():
            parsed_value: Any = PythonEvalUtils.try_expression_evaluation(value, eval_name_space)
            self._additional_values_[key] = parsed_value



    def _parse_components_value_(self, input_data: PipelineConfigurationData, eval_name_space: dict[str, Any]) -> None:
        if input_data.components is None:
            raise ValueError(f"The attribute 'components' of the given '{input_data.__class__.__name__}' is not set "
                             f"although it's required to be set for parsing into '{self.__class__.__name__}'.")

        components = PythonEvalUtils.try_expression_evaluation(input_data.components, eval_name_space)

        if not isinstance(components, DirectionalAcyclicGraph):
            raise TypeError(f"The attribute '{self.__class__.__name__}.components' of this instance was initialized with a wrong type. "
                            f"Expected is 'DirectionalAcyclicGraph[tuple[str, str, dict[str, str]]' but got '{type(components)}'.")

        self._components_ = self._map_input_components_dag_into_internal_components_dag_(components)



    def _validate_attribute_types_(self):
        """
        Geht durch alle Felder von PipelineInputData (Parsed), holt die Typannotation aus PipelineInput.__annotations__,
        und prüft:
          - Wenn Wert nicht None: validate_type(obj, annotation) muss True sein.
          - Wenn None und Annotation kein Optional erlaubt: Exception.
        Für additional-Dicts überspringen wir Typprüfung, da keys und values heterogen sein können.
        """
        type_hints: dict[str, Any] = typing.get_type_hints(PipelineConfiguration)

        for field in fields(PipelineConfigurationData):
            name: str = field.name

            if name != "additional_values":
                transformed_name: str = f"_{name}_"
                self._validate_single_attribute_type(transformed_name, type_hints)



    def _validate_single_attribute_type(self, field_name: str, type_hints: dict[str, Any]) -> None:
        if field_name not in type_hints:
            raise TypeAnnotationError(f"The attribute '{self.__class__.__name__}.{field_name}' is missing type annotations.")

        field_value: Any = getattr(self, field_name)
        type_annotation: Any = type_hints[field_name]

        if field_value is None:
            self._validate_non_set_attribute_type_(field_name, type_annotation)
        else:
            self._validate_set_attribute_type_(field_name, field_value, type_annotation)



    def _validate_set_attribute_type_(self, field_name: str, field_value: Any, type_annotation: Any) -> None:
        if not TypingUtils.does_value_match_type_annotation(field_value, type_annotation):
            raise TypeError(f"The attribute '{self.__class__.__name__}.{field_name}' of this instance was initialized with a wrong type. "
                            f"Expected is '{type_annotation}' but got '{type(field_value)}'.")



    def _validate_non_set_attribute_type_(self, field_name: str, type_annotation: Any) -> None:
        type_origin: typing.ParamSpec = typing.get_origin(type_annotation)
        type_args: tuple = typing.get_args(type_annotation)

        # Checks if type annotation of attributes allows None (e.g. ... | None, Union[..., None], Optional[...])
        # We assume that there is no plain None outside a Union
        if type_origin is typing.Union or type_origin is types.UnionType:
            for argument in type_args:
                if argument is type(None):
                    return

        raise ValueError(f"The attribute '{self.__class__.__name__}.{field_name}' of this instance is not set although it's required to be set.")



    @staticmethod
    def _map_input_components_dag_into_internal_components_dag_(input_dag: DirectionalAcyclicGraph[tuple[str, str, dict[str, str]]]) -> DirectionalAcyclicGraph[PipelineComponentInstantiationInfo]:
        value_mapping: Callable[[tuple[str, str, dict[str, str]]], PipelineComponentInstantiationInfo] = PipelineConfiguration._map_input_dag_node_value_to_instantiation_info_
        return input_dag.value_map(value_mapping)



    @staticmethod
    def _map_input_dag_node_value_to_instantiation_info_(node_value: tuple[str, str, dict[str, str]]) -> PipelineComponentInstantiationInfo:
        from pipeline_entities.pipeline.component_entities.component_registry.component_registry import ComponentRegistry

        component_info: PipelineComponentInfo = ComponentRegistry.get_component(node_value[1])

        if component_info:
            evaluated_overridden_attributes: dict[str, object] = {}

            for attribute_name, attribute_value in node_value[2].items():
                value: object = PythonEvalUtils.try_expression_evaluation(attribute_value, PipelineConfigurationConstants.PARSING_EVAL_NAMESPACE)
                evaluated_overridden_attributes[attribute_name] = value

            return PipelineComponentInstantiationInfo(node_value[0], component_info, evaluated_overridden_attributes)
        else:
            raise NoSuchPipelineComponentError(f"There is no Pipeline component registered with the ID '{node_value[1]}'.")



    # def _map_component_id_tree_to_component_info_tree_(self, component_ids: Tree[str]) -> Tree[PipelineComponentInfo]:
    #     value_mapping: Callable[[str], PipelineComponentInfo] = self._map_component_id_to_meta_info_
    #     return component_ids.value_map(value_mapping)
    #
    #
    #
    # def _map_component_id_to_meta_info_(self, id: str) -> PipelineComponentInfo:
    #     from pipeline_entities.component_entities.component_registry.component_registry import ComponentRegistry
    #
    #     meta_info: PipelineComponentInfo | None = ComponentRegistry.get_component(id)
    #
    #     if meta_info:
    #         return meta_info
    #     else:
    #         raise NoSuchPipelineComponentError(f"There is no Pipeline component registered with the ID '{id}'.")



    def _check_for_static_constraint_violations_(self) -> None:
        for node in self._components_:
            component_info: PipelineComponentInfo = node.value.component

            for static_constraint in component_info.component_meta_info.static_constraints:
                if not static_constraint.evaluate(node, self):
                    raise PipelineConstraintViolationException(f"The static constraint '{static_constraint}' was "
                       f"violated. The violation occurred in the component with the name {repr(node.value.component_name)} "
                       f"and id {repr(component_info.component_id)}.", node, static_constraint)

        return None



    def _check_for_prohibited_override_(self) -> None:
        for node in self._components_:
            instantiation_info: PipelineComponentInstantiationInfo = node.value
            meta_info: ComponentMetaInfo = node.value.component.component_meta_info

            for attribute_name in instantiation_info.overridden_attributes.keys():
                if attribute_name not in meta_info.attributes_allowed_to_be_overridden:
                    raise ProhibitedAttributeOverrideException(f"The attribute {repr(attribute_name)} was overridden "
                       f"in the component {repr(instantiation_info.component_name)} with the component id "
                       f"{repr(instantiation_info.component.component_id)}, but the component id does not allow this override.")





