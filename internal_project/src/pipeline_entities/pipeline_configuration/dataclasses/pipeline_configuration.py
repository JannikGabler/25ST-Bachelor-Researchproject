import math
import types
import typing
from dataclasses import fields
from typing import Any

import jax
import jax.numpy as jnp
import numpy

from data_structures.tree.tree_node import TreeNode
from exceptions.evaluation_error import EvaluationError
from exceptions.type_annotation_error import TypeAnnotationError
from file_handling.dynamic_module_loading.dynamic_module_loader import DynamicModuleLoader

from typing import Callable

from packaging.version import Version

from data_structures.tree.tree import Tree
from exceptions.no_such_pipeline_component_error import NoSuchPipelineComponentError
from pipeline_entities.pipeline_configuration.dataclasses.pipeline_configuration_data import PipelineConfigurationData
from exceptions.pipeline_configuration_constraint_exception import \
    PipelineConfigurationConstraintException
from utils.typing_utils import TypingUtils

from pipeline_entities.component_info.dataclasses.pipeline_component_info import PipelineComponentInfo


class PipelineConfiguration:
    ###########################
    ### Attributes of class ###
    ###########################
    _parsing_eval_namespace_: dict[str, Any] = {'jax': jax, 'jax.numpy': jnp, 'math': math, 'numpy': numpy,
                                                'Version': Version, 'Tree': Tree, 'TreeNode': TreeNode} # Namespace for dynamically loaded modules is getting added on demand



    ###############################
    ### Attributes of instances ###
    ###############################
    _name_: str | None
    _supported_program_version_: Version

    _components_: Tree[PipelineComponentInfo]

    _additional_values_: dict[str, Any]


    ###################
    ### Constructor ###
    ###################
    def __init__(self, input_data: PipelineConfigurationData) -> None:
        self._parse_input_data_(input_data)
        self._validate_attribute_types_()

        self._components_.freeze()

        self._check_for_constraint_violations_()


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
    @property
    def name(self) -> str:
        return self._name_

    @property
    def supported_program_version(self) -> Version:
        return self._supported_program_version_

    @property
    def components(self) -> Tree[PipelineComponentInfo]:
        return self._components_   #__components__ is frozen (immutable)

    @property
    def additional_values(self) -> dict[str, Any]:
        return self._additional_values_



    ##########################
    ### Overridden methods ###
    ##########################
    def __repr__(self):
        return (f"PipelineConfiguration(name={repr(self._name_)}, "
                f"supported_program_version={repr(self._supported_program_version_)}, "
                f"components={repr(self.components)}, "
                f"additional_values={repr(self.additional_values)})")



    def __eq__(self, other):
        if not isinstance(other, PipelineConfiguration):   # Covers None
            return False
        else:
            return self._name_ == other._name_ \
                   and self._supported_program_version_ == other._supported_program_version_ \
                   and self._components_ == other._components_



    def __hash__(self) -> int:   # Instances is immutable
        return hash((self._name_, self._supported_program_version_, self._components_))



    #######################
    ### Private methods ###
    #######################
    def _parse_input_data_(self, input_data: PipelineConfigurationData) -> None:
        eval_name_space: dict[str, Any] = DynamicModuleLoader.get_module_namespace() | self._parsing_eval_namespace_

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
            parsed_value: Any = self._try_expression_evaluation_(field_value, field_name, eval_name_space)
            setattr(self, transformed_field_name, parsed_value)
        else:
            setattr(self, transformed_field_name, None)



    def _parse_additional_input_values_(self, input_data: PipelineConfigurationData, eval_name_space: dict[str, Any]) -> None:
        self._additional_values_ = {}

        for key, value in input_data.additional_values.items():
            parsed_value: Any = self._try_expression_evaluation_(value, key, eval_name_space)
            self._additional_values_[key] = parsed_value



    def _parse_components_value_(self, input_data: PipelineConfigurationData, eval_name_space: dict[str, Any]) -> None:
        if input_data.components is None:
            raise ValueError(f"The attribute '{self.__class__.__name__}.components' of this instance is not set although it's required to be set.")

        components = self._try_expression_evaluation_(input_data.components, "components", eval_name_space)

        if not isinstance(components, Tree):
            raise TypeError(f"The attribute '{self.__class__.__name__}.components' of this instance was initialized with a wrong type. "
                            f"Expected is 'Tree[str]' but got '{type(components)}'.")

        self._components_ = self._map_component_id_tree_to_component_info_tree_(components)



    def _try_expression_evaluation_(self, expression: str, field_name: str, name_space: dict[str, Any]) -> Any:
        try:
            # Security node: __builtins__ are available by default! Might be a security issue (-> define custom safe build ins).
            return eval(expression, {}, name_space)
        except Exception as e:
            raise EvaluationError(f"Error while evaluating '{expression}': {e}")



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



    def _map_component_id_tree_to_component_info_tree_(self, component_ids: Tree[str]) -> Tree[PipelineComponentInfo]:
        value_mapping: Callable[[str], PipelineComponentInfo] = self._map_component_id_to_meta_info_
        return component_ids.value_map(value_mapping)



    def _map_component_id_to_meta_info_(self, id: str) -> PipelineComponentInfo:
        from pipeline_entities.components.dynamic_management.component_registry import ComponentRegistry

        meta_info: PipelineComponentInfo | None = ComponentRegistry.get_component(id)

        if meta_info:
            return meta_info
        else:
            raise NoSuchPipelineComponentError(f"There is no Pipeline component registered with the ID '{id}'.")



    def _check_for_constraint_violations_(self) -> None:
        for tree_node in self._components_:
            component_info: PipelineComponentInfo = tree_node.value

            for static_constraint in component_info.component_meta_info.static_constraints:
                if not static_constraint.evaluate(tree_node, self):
                    raise PipelineConfigurationConstraintException(f"The static constraint '{static_constraint}' of the component"
                       f"'{component_info.component_id}' at the path '{tree_node.path}' was violated.", tree_node, static_constraint)

        return None




