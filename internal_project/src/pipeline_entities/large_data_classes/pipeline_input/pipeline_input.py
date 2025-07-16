import math
import types
import typing
from dataclasses import fields
from typing import Any, Callable, ParamSpec

import jax
import jax.numpy as jnp
import numpy
from packaging.version import Version

from general_data_structures.tree.tree import Tree
from general_data_structures.tree.tree_node import TreeNode
from exceptions.evaluation_error import EvaluationError
from exceptions.type_annotation_error import TypeAnnotationError
from file_handling.dynamic_module_loading.dynamic_module_loader import DynamicModuleLoader
from pipeline_entities.large_data_classes.pipeline_input.pipeline_input_data import PipelineInputData
from utils.typing_utils import TypingUtils


class PipelineInput:
    ###########################
    ### Attributes of class ###
    ###########################
    _parsing_eval_namespace_: dict[str, Any] = {'jax': jax, 'jax.numpy': jnp, 'math': math, 'numpy': numpy,
                                                'Version': Version, 'Tree': Tree, 'TreeNode': TreeNode} # Namespace for dynamically loaded modules is getting added on demand



    ###############################
    ### Attributes of instances ###
    ###############################
    _name_: str | None

    _data_type_: type
    _node_count_: int
    _interpolation_interval_: jnp.ndarray

    _function_expression_: str | None
    _piecewise_function_expression_: list[tuple[tuple[float, float], str]] | None
    _sympy_function_expression_simplification_: bool | None
    _function_callable_: Callable[[jnp.ndarray], jnp.ndarray] | None
    _interpolation_values_: jnp.ndarray | None

    _additional_directly_injected_values_: dict[str, Any]
    _additional_values_: dict[str, Any]



    ###################
    ### Constructor ###
    ###################
    def __init__(self, input_data: PipelineInputData):
        """

        :param input_data:
        """
        self._parse_input_data_(input_data)
        self._validate_attribute_types_()



    #########################
    ### Getters & setters ###
    #########################
    @property
    def name(self) -> str | None:
        return self._name_

    @property
    def data_type(self) -> type:
        return self._data_type_

    @property
    def node_count(self) -> int:
        return self._node_count_

    @property
    def interpolation_interval(self) -> jnp.ndarray:
        return self._interpolation_interval_

    @property
    def function_expression(self) -> str | None:
        return self._function_expression_

    @property
    def piecewise_function_expression(self) -> list[tuple[tuple[any, any], str]] | None:
        return self._piecewise_function_expression_

    @property
    def sympy_function_expression_simplification(self) -> bool | None:
        return self._sympy_function_expression_simplification_

    @property
    def function_callable(self) -> Callable[[jnp.ndarray], jnp.ndarray] | None:
        return self._function_callable_

    @property
    def interpolation_values(self) -> jnp.ndarray | None:
        return self._interpolation_values_

    @property
    def additional_directly_injected_values(self) -> dict[str, Any]:
        return self._additional_directly_injected_values_

    @property
    def additional_values(self) -> dict[str, Any]:
        return self._additional_values_



    ##########################
    ### Overridden methods ###
    ##########################
    def __repr__(self):
        return (f"{self.__class__.__name__}(name='{self._name_}', data_type='{self._data_type_}', "
                f"node_count='{self.node_count}', interpolation_interval='{self.interpolation_interval}', "
                f"function_expression='{self.function_expression}', "
                f"piecewise_function_expressions='{self.piecewise_function_expression}', "
                f"sympy_function_expression_simplification='{self.sympy_function_expression_simplification}', "
                f"function_callable='{self.function_callable}', "
                f"interpolation_values='{self.interpolation_values}', "
                f"additional_directly_injected_values='{self.additional_directly_injected_values}', "
                f"additional_values='{self.additional_values}')")



    #######################
    ### Private methods ###
    #######################
    def _parse_input_data_(self, input_data: PipelineInputData) -> None:
        eval_name_space: dict[str, Any] = DynamicModuleLoader.get_module_namespace() | self._parsing_eval_namespace_

        self._parse_regular_input_values_(input_data, eval_name_space)
        self._parse_additional_input_values_(input_data, eval_name_space)
        self._parse_additional_directly_injected_input_values_(input_data, eval_name_space)



    def _parse_regular_input_values_(self, input_data: PipelineInputData, eval_name_space: dict[str, Any]) -> None:
        for field in fields(PipelineInputData):
            name: str = field.name
            value: str = getattr(input_data, name)

            if name not in ('additional_values', 'additional_directly_injected_values'):
                self._parse_single_regular_input_value_(name, value, eval_name_space)



    def _parse_single_regular_input_value_(self, field_name: str, field_value: str, eval_name_space: dict[str, Any]) -> None:
        transformed_field_name: str = f"_{field_name}_"

        if field_value:
            parsed_value: Any = self._try_expression_evaluation_(field_value, field_name, eval_name_space)
            setattr(self, transformed_field_name, parsed_value)
        else:
            setattr(self, transformed_field_name, None)



    def _parse_additional_input_values_(self, input_data: PipelineInputData, eval_name_space: dict[str, Any]) -> None:
        self._additional_values_ = {}

        for key, value in input_data.additional_values.items():
            parsed_value: Any = self._try_expression_evaluation_(value, key, eval_name_space)
            self._additional_values_[key] = parsed_value



    def _parse_additional_directly_injected_input_values_(self, input_data: PipelineInputData, eval_name_space: dict[str, Any]) -> None:
        self._additional_directly_injected_values_ = {}

        for key, value in input_data.additional_directly_injected_values.items():
            parsed_value: Any = self._try_expression_evaluation_(value, key, eval_name_space)
            self._additional_directly_injected_values_[key] = parsed_value



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
        type_hints: dict[str, Any] = typing.get_type_hints(PipelineInput)

        for field in fields(PipelineInputData):
            name: str = field.name

            if name not in ('additional_values', 'additional_directly_injected_values'):
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
        type_origin: ParamSpec = typing.get_origin(type_annotation)
        type_args: tuple = typing.get_args(type_annotation)

        # Checks if type annotation of attributes allows None (e.g. ... | None, Union[..., None], Optional[...])
        # We assume that there is no plain None outside a Union
        if type_origin is typing.Union or type_origin is types.UnionType:
            for argument in type_args:
                if argument is type(None):
                    return

        raise ValueError(f"The attribute '{self.__class__.__name__}.{field_name}' of this instance is not set although it's required to be set.")