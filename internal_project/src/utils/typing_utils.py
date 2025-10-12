import collections.abc
import types
import typing

import numpy
import jax.numpy
from typing import ParamSpec

from exceptions.not_instantiable_error import NotInstantiableError
from typing import Any


class TypingUtils:
    ###################
    ### Constructor ###
    ###################
    def __init__(self) -> None:
        raise NotInstantiableError(
            f"The class {repr(self.__class__.__name__)} cannot be instantiated."
        )

    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def does_value_match_type_annotation(value: any, type_annotation: any) -> bool:
        """
        Prüft rekursiv, ob obj dem Typ annotation entspricht.
        Unterstützt Union/Optional, list[T], tuple[...] (feste Länge), dict[K, V], Callable, grundlegende Typen und jnp.ndarray.
        Für Any oder wenn annotation is typing.Any: immer True.
        """
        type_origin: ParamSpec = typing.get_origin(type_annotation)
        type_args: tuple = typing.get_args(type_annotation)

        # No type annotation
        if type_annotation is None:
            return True

        # None type annotation
        if type_annotation is type(None):
            return value is None

        # Optional is a Union
        return (
            type_annotation is typing.Any
            or TypingUtils._does_value_match_union_type_annotation_(
                value, type_origin, type_args
            )
            or TypingUtils._does_value_match_list_type_annotation_(
                value, type_origin, type_args
            )
            or TypingUtils._does_value_match_tuple_type_annotation_(
                value, type_origin, type_args
            )
            or TypingUtils._does_value_match_dict_type_annotation_(
                value, type_origin, type_args
            )
            or TypingUtils._does_value_match_callable_type_annotation_(
                value, type_origin
            )
            or (type_origin is not None and isinstance(value, type_origin))
            or (
                isinstance(type_annotation, type) and isinstance(value, type_annotation)
            )
        )  # basic classes (int, str, bool, float, ...)

    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _does_value_match_union_type_annotation_(
        value: Any, type_origin: ParamSpec, type_args: tuple
    ) -> bool:
        if not (type_origin is typing.Union or type_origin is types.UnionType):
            return False

        return any(
            TypingUtils.does_value_match_type_annotation(value, arg)
            for arg in type_args
        )

    @staticmethod
    def _does_value_match_list_type_annotation_(
        value: Any, type_origin: ParamSpec, type_args: tuple
    ) -> bool:
        if not (type_origin in {list, typing.List} and isinstance(value, list)):
            return False
        if not type_args:
            return True

        type_of_elements = type_args[0]

        return all(
            TypingUtils.does_value_match_type_annotation(element, type_of_elements)
            for element in value
        )

    @staticmethod
    def _does_value_match_tuple_type_annotation_(
        value: Any, type_origin: ParamSpec, type_args: tuple
    ) -> bool:
        if not (type_origin in {tuple, typing.Tuple} and isinstance(value, tuple)):
            return False
        if not type_args:
            return True

        # Handle tuples with a variable length (-> tuple[T, ...])
        if len(type_args) == 2 and type_args[1] is Ellipsis:
            type_of_elements = type_args[0]
            return all(
                TypingUtils.does_value_match_type_annotation(element, type_of_elements)
                for element in value
            )

        # Handle tuples with a fixed length (e.g. tuple[int, int])
        return len(value) == len(type_args) and all(
            TypingUtils.does_value_match_type_annotation(elem, ann)
            for elem, ann in zip(value, type_args)
        )

    @staticmethod
    def _does_value_match_dict_type_annotation_(
        value: Any, type_origin: ParamSpec, type_args: tuple
    ) -> bool:
        if not (type_origin in {dict, typing.Dict} and isinstance(value, dict)):
            return False
        if not type_args or type_args == (Any, Any):
            return True

        key_type, val_type = type_args

        return all(
            TypingUtils.does_value_match_type_annotation(k, key_type)
            and TypingUtils.does_value_match_type_annotation(v, val_type)
            for k, v in value.items()
        )

    @staticmethod
    def _does_value_match_callable_type_annotation_(
        value: Any, type_origin: ParamSpec
    ) -> bool:
        return type_origin in {
            callable,
            typing.Callable,
            collections.abc.Callable,
        } and callable(
            value
        )  # Right parameter and return type cannot be checked

    @staticmethod
    def _does_value_match_array_type_annotation_(
        value: Any, type_annotation: Any
    ) -> bool:
        return (
            type_annotation is numpy.ndarray and isinstance(value, numpy.ndarray)
        ) or (
            type_annotation is jax.numpy.ndarray
            and isinstance(value, jax.numpy.ndarray)
        )
