import math
import types
import typing
from dataclasses import fields
from typing import Callable, ParamSpec

import jax
import jax.numpy as jnp
import numpy
from jax.typing import DTypeLike
from packaging.version import Version

from general_data_structures.tree.tree import Tree
from general_data_structures.tree.tree_node import TreeNode
from exceptions.evaluation_error import EvaluationError
from exceptions.type_annotation_error import TypeAnnotationError
from file_handling.dynamic_module_loading.dynamic_module_loader import DynamicModuleLoader
from data_classes.pipeline_input.pipeline_input_data import PipelineInputData
from utils.typing_utils import TypingUtils


class PipelineInput:
    """
    Parsed pipeline input container. Stores typed function definitions, interpolation settings,
    evaluation points, and additional values after parsing and optional expression evaluation.

    Raises:
        EvaluationError: If evaluation of an expression fails.
        TypeError: If an attribute of an instance is initialized with a wrong type.
        TypeAnnotationError: If a required attribute is missing type annotations.
        ValueError: If the attribute of an instance is not set although it is required to be set.
    """


    ###########################
    ### Attributes of class ###
    ###########################
    _parsing_eval_namespace_: dict[str, object] = {
        "jax": jax,
        "jax.numpy": jnp,
        "math": math,
        "numpy": numpy,
        "Version": Version,
        "Tree": Tree,
        "TreeNode": TreeNode,
    }


    ###############################
    ### Attributes of instances ###
    ###############################
    _name_: str | None

    _data_type_: DTypeLike
    _node_count_: int
    _interpolation_interval_: jnp.ndarray

    _function_expression_: str | None
    _piecewise_function_expression_: list[tuple[tuple[float, float], str]] | None
    _sympy_function_expression_simplification_: bool | None
    _function_callable_: Callable[[jnp.ndarray], jnp.ndarray] | None
    _interpolation_values_: jnp.ndarray | None

    _interpolant_evaluation_points_: jnp.ndarray | None

    _additional_directly_injected_values_: dict[str, object]
    _additional_values_: dict[str, object]


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
        """
        The name of the pipeline input.

        Returns:
            str: Pipeline name.
        """

        return self._name_


    @property
    def data_type(self) -> DTypeLike:
        """
        The data type.

        Returns:
            DTypeLike: Data type.
        """

        return self._data_type_


    @property
    def node_count(self) -> int:
        """
        The number of interpolation nodes.

        Returns:
            int: Node count.
        """

        return self._node_count_


    @property
    def interpolation_interval(self) -> jnp.ndarray:
        """
        The interpolation interval.

        Returns:
            jnp.ndarray: Interval boundaries as an array.
        """

        return self._interpolation_interval_


    @property
    def function_expression(self) -> str | None:
        """
        The function expression as a string.

        Returns:
            str | None: Expression string or None if not set.
        """

        return self._function_expression_


    @property
    def piecewise_function_expression(self) -> list[tuple[tuple[any, any], str]] | None:
        """
        The piecewise function expression.

        Returns:
            list[tuple[tuple[float, float], str]] | None: List of interval-expression pairs or None if not set.
        """

        return self._piecewise_function_expression_


    @property
    def sympy_function_expression_simplification(self) -> bool | None:
        """
        Whether SymPy simplification should be applied.

        Returns:
            bool | None: True/False if set, otherwise None.
        """

        return self._sympy_function_expression_simplification_


    @property
    def function_callable(self) -> Callable[[jnp.ndarray], jnp.ndarray] | None:
        """
        Callable version of the function.

        Returns:
            Callable[[jnp.ndarray], jnp.ndarray] | None: Callable function or None if not set.
        """

        return self._function_callable_


    @property
    def interpolation_values(self) -> jnp.ndarray | None:
        """
        The interpolation values.

        Returns:
            jnp.ndarray | None: Values or None if not set.
        """

        return self._interpolation_values_


    @property
    def interpolant_evaluation_points(self):
        """
        The points at which the interpolant is evaluated.

        Returns:
            jnp.ndarray | None: Evaluation points or None if not set.
        """

        return self._interpolant_evaluation_points_


    @property
    def additional_directly_injected_values(self) -> dict[str, object]:
        """
        Additional injected values.

        Returns:
            dict[str, object]: Additional injected values.
        """

        return self._additional_directly_injected_values_


    @property
    def additional_values(self) -> dict[str, object]:
        """
        Additional evaluated values.

        Returns:
            dict[str, object]: Additional evaluated values.
        """

        return self._additional_values_


    ##########################
    ### Overridden methods ###
    ##########################
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={repr(self._name_)}', data_type='{repr(self._data_type_)}', "
            f"node_count={repr(self.node_count)}, interpolation_interval={repr(self.interpolation_interval)}, "
            f"function_expression={repr(self.function_expression)}, "
            f"piecewise_function_expressions={repr(self.piecewise_function_expression)}, "
            f"sympy_function_expression_simplification='{repr(self.sympy_function_expression_simplification)}', "
            f"function_callable={repr(self.function_callable)}, "
            f"interpolation_values={repr(self.interpolation_values)}', "
            f"interpolant_evaluation_points={repr(self.interpolant_evaluation_points)}, "
            f"additional_directly_injected_values={repr(self.additional_directly_injected_values)}, "
            f"additional_values={repr(self.additional_values)}')"
        )


    def __str__(self) -> str:
        return self.__repr__()


    def __hash__(self) -> int:
        return hash(
            (
                self._name_,
                self._data_type_,
                self._node_count_,
                self._interpolation_interval_,
                self._function_expression_,
                self._piecewise_function_expression_,
                self._sympy_function_expression_simplification_,
                self._function_callable_,
                self._interpolation_values_,
                self._interpolant_evaluation_points_,
                self._additional_directly_injected_values_,
                self._additional_values_,
            )
        )


    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        else:
            return (
                self._name_ == other._name_
                and self._data_type_ == other._data_type_
                and self._node_count_ == other._node_count_
                and jnp.array_equal(
                    self._interpolation_interval_, other._interpolation_interval_
                )
                and self._function_expression_ == other._function_expression_
                and self._piecewise_function_expression_
                == other._piecewise_function_expression_
                and self._sympy_function_expression_simplification_
                == other._sympy_function_expression_simplification_
                and (
                    self._function_callable_
                    is None
                    == other._function_callable_
                    is None
                )
                and jnp.array_equal(
                    self._interpolation_values_, other._interpolation_values_
                )
                and jnp.array_equal(
                    self._interpolant_evaluation_points_,
                    other._interpolant_evaluation_points_,
                )
                and self._additional_directly_injected_values_
                == other._additional_directly_injected_values_
                and self._additional_values_ == other._additional_values_
            )


    #######################
    ### Private methods ###
    #######################
    def _parse_input_data_(self, input_data: PipelineInputData) -> None:
        eval_name_space: dict[str, object] = (DynamicModuleLoader.get_module_namespace() | self._parsing_eval_namespace_)

        self._parse_regular_input_values_(input_data, eval_name_space)
        self._parse_additional_input_values_(input_data, eval_name_space)
        self._parse_additional_directly_injected_input_values_(input_data, eval_name_space)


    def _parse_regular_input_values_(self, input_data: PipelineInputData, eval_name_space: dict[str, object]) -> None:
        for field in fields(PipelineInputData):
            name: str = field.name
            value: str = getattr(input_data, name)

            if name not in ("additional_values", "additional_directly_injected_values"):
                self._parse_single_regular_input_value_(name, value, eval_name_space)


    def _parse_single_regular_input_value_(self, field_name: str, field_value: str, eval_name_space: dict[str, object]) -> None:
        transformed_field_name: str = f"_{field_name}_"

        if field_value:
            parsed_value: object = self._try_expression_evaluation_(field_value, field_name, eval_name_space)
            setattr(self, transformed_field_name, parsed_value)
        else:
            setattr(self, transformed_field_name, None)


    def _parse_additional_input_values_(self, input_data: PipelineInputData, eval_name_space: dict[str, object]) -> None:
        self._additional_values_ = {}

        for key, value in input_data.additional_values.items():
            parsed_value: object = self._try_expression_evaluation_(value, key, eval_name_space)
            self._additional_values_[key] = parsed_value


    def _parse_additional_directly_injected_input_values_(self, input_data: PipelineInputData, eval_name_space: dict[str, object]) -> None:
        self._additional_directly_injected_values_ = {}

        for key, value in input_data.additional_directly_injected_values.items():
            parsed_value: object = self._try_expression_evaluation_(value, key, eval_name_space)
            self._additional_directly_injected_values_[key] = parsed_value


    @staticmethod
    def _try_expression_evaluation_(expression: str, field_name: str, name_space: dict[str, object]) -> object:
        try:
            return eval(expression, {}, name_space)
        except Exception as e:
            raise EvaluationError(f"Error while evaluating '{expression}': {e}")


    def _validate_attribute_types_(self) -> None:
        type_hints: dict[str, object] = typing.get_type_hints(PipelineInput)

        for field in fields(PipelineInputData):
            name: str = field.name

            if name not in ("additional_values", "additional_directly_injected_values"):
                transformed_name: str = f"_{name}_"
                self._validate_single_attribute_type(transformed_name, type_hints)


    def _validate_single_attribute_type(self, field_name: str, type_hints: dict[str, object]) -> None:
        if field_name not in type_hints:
            raise TypeAnnotationError(f"The attribute '{self.__class__.__name__}.{field_name}' is missing type annotations.")

        field_value: object = getattr(self, field_name)
        type_annotation: object = type_hints[field_name]

        if field_value is None:
            self._validate_non_set_attribute_type_(field_name, type_annotation)
        else:
            self._validate_set_attribute_type_(field_name, field_value, type_annotation)


    def _validate_set_attribute_type_(self, field_name: str, field_value: object, type_annotation: object) -> None:
        if not TypingUtils.does_value_match_type_annotation(field_value, type_annotation):
            raise TypeError(
                f"The attribute '{self.__class__.__name__}.{field_name}' of this instance was initialized with a wrong type. "
                f"Expected is '{type_annotation}' but got '{type(field_value)}'."
            )


    def _validate_non_set_attribute_type_(self, field_name: str, type_annotation: object) -> None:
        type_origin: ParamSpec = typing.get_origin(type_annotation)
        type_args: tuple = typing.get_args(type_annotation)

        if type_origin is typing.Union or type_origin is types.UnionType:
            for argument in type_args:
                if argument is type(None):
                    return

        raise ValueError(f"The attribute '{self.__class__.__name__}.{field_name}' of this instance is not set although it's required to be set.")
