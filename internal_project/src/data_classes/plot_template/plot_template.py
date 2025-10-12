from __future__ import annotations

from abc import abstractmethod, ABC
from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt
import types
import weakref


class WrapperBase(ABC):
    _owner_: PlotTemplate
    _log_: list

    def __init__(self, owner: PlotTemplate) -> None:
        self._owner_ = owner
        self._log_ = []

    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            return super().__getattribute__(name)

        return self._get_attr_of_internal_obj_(name)

    def __setattr__(self, name, value) -> None:
        if name.startswith("_"):
            # internal variables of wrapper itself
            super().__setattr__(name, value)
        else:
            self._owner_._fix_internal_state_()
            internal_obj = self._get_internal_object_()
            self._log_.append(("setattr", name, deepcopy(value)))
            setattr(internal_obj, name, value)

    def _get_attr_of_internal_obj_(self, name: str) -> object:
        self._owner_._fix_internal_state_()
        internal_obj = self._get_internal_object_()
        attr = getattr(internal_obj, name)

        # Method?
        if isinstance(attr, types.MethodType):

            def wrapper(*args, **kwargs):
                if name not in ["show", "savefig"]:
                    self._log_.append(("call", name, deepcopy(args), deepcopy(kwargs)))
                return attr(*args, **kwargs)

            return wrapper

        return attr

    def _replay_operations_on_internal_object_(self) -> None:
        """Spielt die geloggten Operationen auf ein anderes Objekt ab"""
        internal_obj = self._get_internal_object_()
        for op, name, *rest in self._log_:
            if op == "call":
                args, kwargs = rest
                getattr(internal_obj, name)(*args, **kwargs)
            elif op == "setattr":
                (value,) = rest
                setattr(internal_obj, name, value)

    @abstractmethod
    def _get_internal_object_(self) -> plt.Figure | plt.Axes:
        pass


class FigureWrapper(WrapperBase):
    # Figure-sided comfort methods
    def show(self, *args, **kwargs):
        """Delegiert an fig.show(...)."""
        super()._get_attr_of_internal_obj_("show")(*args, **kwargs)

    def savefig(self, *args, **kwargs):
        """Delegiert an fig.savefig(...)."""
        super()._get_attr_of_internal_obj_("savefig")(*args, **kwargs)

    def suptitle(self, *args, **kwargs):
        """Delegiert an fig.suptitle(...)."""
        super()._get_attr_of_internal_obj_("suptitle")(*args, **kwargs)

    def _get_internal_object_(self) -> plt.Figure:
        return PlotTemplate._internal_figure_axes_dict_[self._owner_][0]


class AxesWrapper(WrapperBase):
    def plot(self, *args, **kwargs):
        """Delegiert an ax.plot(...)."""
        super()._get_attr_of_internal_obj_("plot")(*args, **kwargs)

    def scatter(self, *args, **kwargs):
        """Delegiert an ax.scatter(...)."""
        super()._get_attr_of_internal_obj_("scatter")(*args, **kwargs)

    def set_title(self, *args, **kwargs):
        """Delegiert an ax.set_title(...)."""
        super()._get_attr_of_internal_obj_("set_title")(*args, **kwargs)

    def set_xlabel(self, *args, **kwargs):
        """Delegiert an ax.set_xlabel(...)."""
        super()._get_attr_of_internal_obj_("set_xlabel")(*args, **kwargs)

    def set_ylabel(self, *args, **kwargs):
        """Delegiert an ax.set_ylabel(...)."""
        super()._get_attr_of_internal_obj_("set_ylabel")(*args, **kwargs)

    def grid(self, *args, **kwargs):
        """Delegiert an ax.legend(...)."""
        super()._get_attr_of_internal_obj_("grid")(*args, **kwargs)

    def legend(self, *args, **kwargs):
        """Delegiert an ax.legend(...)."""
        super()._get_attr_of_internal_obj_("legend")(*args, **kwargs)

    def _get_internal_object_(self) -> plt.Axes:
        return PlotTemplate._internal_figure_axes_dict_[self._owner_][1]


class PlotTemplate:
    _internal_figure_axes_dict_: dict[object, tuple[plt.Figure, plt.Axes]] = (
        weakref.WeakKeyDictionary()
    )

    _subplots_args_: tuple[Any, ...]
    _subplots_kwargs_: dict[str, Any]

    _external_fig_: FigureWrapper
    _external_axes_: AxesWrapper

    def __init__(self, *args, **kwargs):
        self._subplots_args_ = deepcopy(args)
        self._subplots_kwargs_ = deepcopy(kwargs)
        self._external_fig_ = FigureWrapper(self)
        self._external_axes_ = AxesWrapper(self)

    @property
    def fig(self) -> FigureWrapper:
        return self._external_fig_

    @property
    def ax(self) -> AxesWrapper:
        return self._external_axes_

    def _fix_internal_state_(self) -> None:
        if self not in self._internal_figure_axes_dict_:
            fig, ax = plt.subplots(*self._subplots_args_, **self._subplots_kwargs_)
            self._internal_figure_axes_dict_[self] = (fig, ax)
            self._external_axes_._replay_operations_on_internal_object_()
            self._external_fig_._replay_operations_on_internal_object_()

    # def __repr__(self) -> str:
    #     return (f"{self.__class__.__name__}(subplots_args={repr(self._subplots_args_)}, subplots_kwargs={repr(self._subplots_kwargs_)}, "
    #             f"external_fig={repr(self._external_fig_)}, external_axes={repr(self._external_axes_)})")
    #
    # def __str__(self) -> str:
    #     return (f"{self.__class__.__name__}(subplots_args={str(self._subplots_args_)}, subplots_kwargs={str(self._subplots_kwargs_)}, "
    #             f"external_fig={str(self._external_fig_)}, external_axes={str(self._external_axes_)})")
