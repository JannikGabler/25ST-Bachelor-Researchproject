import collections.abc
import types
import typing

import numpy
import jax.numpy
from typing import ParamSpec

from exceptions.not_instantiable_error import NotInstantiableError
from typing import Any


class TypingUtils:
    """
    Utility helpers for runtime checks against typing annotations. This class is not meant to be instantiated.
    """


    ###################
    ### Constructor ###
    ###################
    def __init__(self) -> None:
        """
        Raises:
            NotInstantiableError: Always raised when initialized to indicate that this class is not meant to be instantiated.
        """

        raise NotInstantiableError(f"The class {repr(self.__class__.__name__)} cannot be instantiated.")


    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def does_value_match_type_annotation(value: any, type_annotation: any) -> bool:
        """
        Recursively check whether the given value matches the given typing annotation.
        Supports Union/Optional, list[T], tuple[...] (fixed or variable length), dict[K, V], Callable, basic classes,
        and array types. Returns True for Any/typing.Any.

        Args:
            value (any): The value to validate.
            type_annotation (any): The typing annotation to validate against.

        Returns:
            bool: True if the value matches the annotation, otherwise False.
        """

        type_origin: ParamSpec = typing.get_origin(type_annotation)
        type_args: tuple = typing.get_args(type_annotation)

        if type_annotation is None:
            return True

        if type_annotation is type(None):
            return value is None

        return (type_annotation is typing.Any or TypingUtils._does_value_match_union_type_annotation_(value, type_origin, type_args)
            or TypingUtils._does_value_match_list_type_annotation_(value, type_origin, type_args)
            or TypingUtils._does_value_match_tuple_type_annotation_(value, type_origin, type_args)
            or TypingUtils._does_value_match_dict_type_annotation_(value, type_origin, type_args)
            or TypingUtils._does_value_match_callable_type_annotation_(value, type_origin)
            or (type_origin is not None and isinstance(value, type_origin))
            or (isinstance(type_annotation, type) and isinstance(value, type_annotation)))


    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _does_value_match_union_type_annotation_(value: Any, type_origin: ParamSpec, type_args: tuple) -> bool:
        if not (type_origin is typing.Union or type_origin is types.UnionType):
            return False

        return any(TypingUtils.does_value_match_type_annotation(value, arg) for arg in type_args)


    @staticmethod
    def _does_value_match_list_type_annotation_(value: Any, type_origin: ParamSpec, type_args: tuple) -> bool:
        if not (type_origin in {list, typing.List} and isinstance(value, list)):
            return False
        if not type_args:
            return True

        type_of_elements = type_args[0]

        return all(TypingUtils.does_value_match_type_annotation(element, type_of_elements) for element in value)


    @staticmethod
    def _does_value_match_tuple_type_annotation_(value: Any, type_origin: ParamSpec, type_args: tuple) -> bool:
        if not (type_origin in {tuple, typing.Tuple} and isinstance(value, tuple)):
            return False
        if not type_args:
            return True

        if len(type_args) == 2 and type_args[1] is Ellipsis:
            type_of_elements = type_args[0]
            return all(TypingUtils.does_value_match_type_annotation(element, type_of_elements) for element in value)

        return len(value) == len(type_args) and all(TypingUtils.does_value_match_type_annotation(elem, ann) for elem, ann in zip(value, type_args))


    @staticmethod
    def _does_value_match_dict_type_annotation_(value: Any, type_origin: ParamSpec, type_args: tuple) -> bool:
        if not (type_origin in {dict, typing.Dict} and isinstance(value, dict)):
            return False
        if not type_args or type_args == (Any, Any):
            return True

        key_type, val_type = type_args

        return all(TypingUtils.does_value_match_type_annotation(k, key_type) and TypingUtils.does_value_match_type_annotation(v, val_type) for k, v in value.items())


    @staticmethod
    def _does_value_match_callable_type_annotation_(value: Any, type_origin: ParamSpec) -> bool:
        return type_origin in {callable, typing.Callable, collections.abc.Callable,} and callable(value)


    @staticmethod
    def _does_value_match_array_type_annotation_(value: Any, type_annotation: Any) -> bool:
        return ((type_annotation is numpy.ndarray and isinstance(value, numpy.ndarray))
             or (type_annotation is jax.numpy.ndarray and isinstance(value, jax.numpy.ndarray)))
