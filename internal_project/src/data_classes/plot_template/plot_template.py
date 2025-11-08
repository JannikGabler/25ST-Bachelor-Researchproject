from __future__ import annotations

from abc import abstractmethod, ABC
from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt
import types
import weakref


class WrapperBase(ABC):
    """
    Abstract base class for plot object wrappers.
    Manages delegation of attribute access and method calls to an internal Matplotlib object while recording all
    operations for later replay or synchronization.
    """


    _owner_: PlotTemplate
    _log_: list


    def __init__(self, owner: PlotTemplate) -> None:
        """
        Args:
            owner (PlotTemplate): The PlotTemplate instance that owns this wrapper and provides access to the underlying plotting object.

        Returns:
            None
        """

        self._owner_ = owner
        self._log_ = []


    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            return super().__getattribute__(name)

        return self._get_attr_of_internal_obj_(name)


    def __setattr__(self, name, value) -> None:
        if name.startswith("_"):
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

        if isinstance(attr, types.MethodType):

            def wrapper(*args, **kwargs):
                if name not in ["show", "savefig"]:
                    self._log_.append(("call", name, deepcopy(args), deepcopy(kwargs)))
                return attr(*args, **kwargs)

            return wrapper

        return attr


    def _replay_operations_on_internal_object_(self) -> None:
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
    """
    Wrapper class for Matplotlib Figure objects.
    Provides controlled access to the underlying figure while logging all operations for reproducibility and
    synchronization within a PlotTemplate.
    """

    def show(self, *args, **kwargs):
        """
        Delegate to the internal figure's show(...) method.

        Args:
            *args: Positional arguments passed to `matplotlib.figure.Figure.show`.
            **kwargs: Keyword arguments passed to `matplotlib.figure.Figure.show`.

        Returns:
            None
        """

        super()._get_attr_of_internal_obj_("show")(*args, **kwargs)


    def savefig(self, *args, **kwargs):
        """
        Delegate to the internal figure's savefig(...) method.

        Args:
            *args: Positional arguments passed to `matplotlib.figure.Figure.savefig`.
            **kwargs: Keyword arguments passed to `matplotlib.figure.Figure.savefig`.

        Returns:
            None
        """

        super()._get_attr_of_internal_obj_("savefig")(*args, **kwargs)


    def suptitle(self, *args, **kwargs):
        """
        Delegate to the internal figure's suptitle(...) method.

        Args:
            *args: Positional arguments passed to `matplotlib.figure.Figure.suptitle`.
            **kwargs: Keyword arguments passed to `matplotlib.figure.Figure.suptitle`.

        Returns:
            None
        """

        super()._get_attr_of_internal_obj_("suptitle")(*args, **kwargs)


    def _get_internal_object_(self) -> plt.Figure:
        return PlotTemplate._internal_figure_axes_dict_[self._owner_][0]



class AxesWrapper(WrapperBase):
    """
    Wrapper class for Matplotlib Axes objects.
    Provides controlled access to the underlying axes while logging all operations for reproducibility and synchronization within a PlotTemplate.
    """

    def plot(self, *args, **kwargs):
        """
        Delegate to the internal axes' plot(...) method.

        Args:
            *args: Positional arguments passed to `matplotlib.axes.Axes.plot`.
            **kwargs: Keyword arguments passed to `matplotlib.axes.Axes.plot`.

        Returns:
            None
        """

        super()._get_attr_of_internal_obj_("plot")(*args, **kwargs)


    def scatter(self, *args, **kwargs):
        """
        Delegate to the internal axes' scatter(...) method.

        Args:
            *args: Positional arguments passed to `matplotlib.axes.Axes.scatter`.
            **kwargs: Keyword arguments passed to `matplotlib.axes.Axes.scatter`.

        Returns:
            None
        """

        super()._get_attr_of_internal_obj_("scatter")(*args, **kwargs)


    def set_title(self, *args, **kwargs):
        """
        Delegate to the internal axes' set_title(...) method.

        Args:
            *args: Positional arguments passed to `matplotlib.axes.Axes.set_title`.
            **kwargs: Keyword arguments passed to `matplotlib.axes.Axes.set_title`.

        Returns:
            None
        """

        super()._get_attr_of_internal_obj_("set_title")(*args, **kwargs)


    def set_xlabel(self, *args, **kwargs):
        """
        Delegate to the internal axes' set_xlabel(...) method.

        Args:
            *args: Positional arguments passed to `matplotlib.axes.Axes.set_xlabel`.
            **kwargs: Keyword arguments passed to `matplotlib.axes.Axes.set_xlabel`.

        Returns:
            None
        """

        super()._get_attr_of_internal_obj_("set_xlabel")(*args, **kwargs)


    def set_ylabel(self, *args, **kwargs):
        """
        Delegate to the internal axes' set_ylabel(...) method.

        Args:
            *args: Positional arguments passed to `matplotlib.axes.Axes.set_ylabel`.
            **kwargs: Keyword arguments passed to `matplotlib.axes.Axes.set_ylabel`.

        Returns:
            None
        """

        super()._get_attr_of_internal_obj_("set_ylabel")(*args, **kwargs)


    def grid(self, *args, **kwargs):
        """
        Delegate to the internal axes' grid(...) method.

        Args:
            *args: Positional arguments passed to `matplotlib.axes.Axes.grid`.
            **kwargs: Keyword arguments passed to `matplotlib.axes.Axes.grid`.

        Returns:
            None
        """

        super()._get_attr_of_internal_obj_("grid")(*args, **kwargs)


    def legend(self, *args, **kwargs):
        """
        Delegate to the internal axes' legend(...) method.

        Args:
            *args: Positional arguments passed to `matplotlib.axes.Axes.legend`.
            **kwargs: Keyword arguments passed to `matplotlib.axes.Axes.legend`.

        Returns:
            None
        """

        super()._get_attr_of_internal_obj_("legend")(*args, **kwargs)


    def _get_internal_object_(self) -> plt.Axes:
        return PlotTemplate._internal_figure_axes_dict_[self._owner_][1]


class PlotTemplate:
    """
    Template class for creating and managing Matplotlib figure–axes pairs.
    Provides external wrapper objects for controlled access and logs all
    operations on the figure and axes for reproducibility and state restoration.
    """

    _internal_figure_axes_dict_: dict[object, tuple[plt.Figure, plt.Axes]] = (weakref.WeakKeyDictionary())

    _subplots_args_: tuple[Any, ...]
    _subplots_kwargs_: dict[str, Any]

    _external_fig_: FigureWrapper
    _external_axes_: AxesWrapper


    def __init__(self, *args, **kwargs):
        """
        Args:
            *args: Positional arguments passed to `matplotlib.pyplot.subplots`.
            **kwargs: Keyword arguments passed to `matplotlib.pyplot.subplots`.

        Returns:
            None
        """

        self._subplots_args_ = deepcopy(args)
        self._subplots_kwargs_ = deepcopy(kwargs)
        self._external_fig_ = FigureWrapper(self)
        self._external_axes_ = AxesWrapper(self)


    @property
    def fig(self) -> FigureWrapper:
        """
        Provides access to the wrapped Matplotlib Figure instance.

        Returns:
            FigureWrapper: Wrapper providing controlled access to the figure.
        """

        return self._external_fig_


    @property
    def ax(self) -> AxesWrapper:
        """
        Provides access to the wrapped Matplotlib Axes instance.

        Returns:
            AxesWrapper: Wrapper providing controlled access to the axes.
        """

        return self._external_axes_


    def _fix_internal_state_(self) -> None:
        if self not in self._internal_figure_axes_dict_:
            fig, ax = plt.subplots(*self._subplots_args_, **self._subplots_kwargs_)
            self._internal_figure_axes_dict_[self] = (fig, ax)
            self._external_axes_._replay_operations_on_internal_object_()
            self._external_fig_._replay_operations_on_internal_object_()

