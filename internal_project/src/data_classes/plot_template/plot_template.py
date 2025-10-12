from __future__ import annotations

from abc import abstractmethod, ABC
from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt
import types

# from typing import Any, Callable, List, Tuple, Dict
# import matplotlib.pyplot as plt
#
# class _Op:
#     """Repräsentiert eine aufgezeichnete Operation."""
#     def __init__(self, kind: str, name: str, args: Tuple, kwargs: Dict):
#         # kind: 'call' oder 'setattr'
#         self.kind = kind
#         self.name = name
#         self.args = args
#         self.kwargs = kwargs
#
#     def __repr__(self):
#         return f"_Op({self.kind!r}, {self.name!r}, args={self.args}, kwargs={self.kwargs})"
#
# class DeferredResult:
#     """Platzhalter für Rückgabewerte von Methoden (nur Minimalfunktionalität)."""
#     def __init__(self, desc: str = "<deferred>"):
#         self._desc = desc
#
#     def __repr__(self):
#         return f"<DeferredResult {self._desc}>"
#
# class _TargetProxy:
#     """
#     Proxy für fig oder ax. Jede Methode/Attributzuweisung wird aufgezeichnet.
#     Interne Namen müssen mit '_' beginnen, damit sie auf das Proxy-Objekt selbst gesetzt werden.
#     """
#     def __init__(self, owner: 'PlotTemplate', target_name: str):
#         # owner hält die Listen fig_ops bzw. ax_ops
#         object.__setattr__(self, "_owner", owner)
#         object.__setattr__(self, "_target_name", target_name)  # 'fig' oder 'ax'
#
#     def __getattr__(self, name: str) -> Callable:
#         # Wenn auf eine nicht-existente Methode/Attribut zugegriffen wird, geben wir eine Funktion zurück,
#         # die die Aufruf-Argumente aufzeichnet.
#         def recorder(*args, **kwargs):
#             self._owner._record(self._target_name, _Op('call', name, args, kwargs))
#             # Wir geben ein DeferredResult zurück — kein echtes Artist-Objekt.
#             return DeferredResult(f"{self._target_name}.{name}()")
#         return recorder
#
#     def __setattr__(self, name: str, value: Any):
#         # interne Attribute direkt setzen
#         if name.startswith("_"):
#             object.__setattr__(self, name, value)
#         else:
#             # Aufzeichnung der Attributzuweisung
#             self._owner._record(self._target_name, _Op('setattr', name, (value,), {}))
#
# class PlotTemplate:
#     """
#     Hauptklasse. Verwende so:
#         wrapper = DeferredSubplot(figsize=(6,4))
#         wrapper.ax.plot([1,2,3], [1,4,9])
#         wrapper.ax.set_title("Deferred")
#         wrapper.fig.suptitle("My plot")
#         wrapper.savefig("out.png")
#     """
#     def __init__(self, *subplots_args, **subplots_kwargs):
#         """
#         subplots_args/kwargs werden später an plt.subplots(...) gegeben,
#         wenn das Replay ausgeführt wird. Du kannst hier z. B. figsize=(6,4) setzen.
#         """
#         self._subplots_args = subplots_args
#         self._subplots_kwargs = subplots_kwargs
#
#         # Listen der aufgezeichneten Operationen
#         self._fig_ops: List[_Op] = []
#         self._ax_ops: List[_Op] = []
#
#         # Proxies, die der Benutzer benutzt
#         self.fig = _TargetProxy(self, 'fig')
#         self.ax = _TargetProxy(self, 'ax')
#
#         # Flag ob bereits ein reales fig/ax erzeugt wurde (bei replay)
#         self._replayed_real = None  # Optional[Tuple[Figure, Axes]]
#
#     def _record(self, target: str, op: _Op):
#         if target == 'fig':
#             self._fig_ops.append(op)
#         elif target == 'ax':
#             self._ax_ops.append(op)
#         else:
#             raise ValueError("target must be 'fig' or 'ax'")
#
#     def replay(self):
#         """
#         Erzeuge ein echtes fig, ax (plt.subplots mit den beim Konstruktor
#         gegebenen Args/Kwargs) und wende alle aufgezeichneten Operationen an.
#         Liefert das tatsächliche (fig, ax)-Tupel zurück.
#         """
#         fig, ax = plt.subplots(*self._subplots_args, **self._subplots_kwargs)
#         # Wende fig-Operationen an
#         for op in self._fig_ops:
#             if op.kind == 'call':
#                 func = getattr(fig, op.name, None)
#                 if func is None:
#                     raise AttributeError(f"Figure hat kein Attribut {op.name!r}")
#                 func(*op.args, **op.kwargs)
#             elif op.kind == 'setattr':
#                 setattr(fig, op.name, op.args[0])
#         # Wende ax-Operationen an
#         for op in self._ax_ops:
#             if op.kind == 'call':
#                 func = getattr(ax, op.name, None)
#                 if func is None:
#                     raise AttributeError(f"Axes hat kein Attribut {op.name!r}")
#                 func(*op.args, **op.kwargs)
#             elif op.kind == 'setattr':
#                 setattr(ax, op.name, op.args[0])
#         # speichere Referenz
#         self._replayed_real = (fig, ax)
#         return fig, ax
#
#     def show(self, *args, **kwargs):
#         """
#         Recreate and show. Gibt das reale (fig, ax) zurück.
#         Zusätzliche args/kwargs werden an plt.show() weitergereicht.
#         """
#         fig, ax = self.replay()
#         plt.show(*args, **kwargs)
#         return fig, ax
#
#     def savefig(self, *args, **kwargs):
#         """
#         Replay und savefig auf dem wiederhergestellten figure ausführen.
#         Beispiel: wrapper.savefig("out.png", dpi=150)
#         """
#         fig, ax = self.replay()
#         fig.savefig(*args, **kwargs)
#         return fig, ax
#
#     def clear_recordings(self):
#         """Löscht alle aufgezeichneten Operationen (nicht das bereits gerenderte fig/ax)."""
#         self._fig_ops.clear()
#         self._ax_ops.clear()
#         self._replayed_real = None
#
#     def get_recorded_ops(self):
#         """Hilfsfunktion zur Inspektion im Debugging."""
#         return {'fig': list(self._fig_ops), 'ax': list(self._ax_ops)}
import weakref
# from __future__ import annotations
#
# from typing import Any, Callable, List, Tuple, Dict, Optional
# import matplotlib.pyplot as plt
# from matplotlib.figure import Figure
# from matplotlib.axes import Axes
#
#
# class _Op:
#     """Repräsentiert eine aufgezeichnete Operation."""
#     def __init__(self, kind: str, name: str, args: Tuple, kwargs: Dict):
#         # kind: 'call' oder 'setattr'
#         self.kind = kind
#         self.name = name
#         self.args = args
#         self.kwargs = kwargs
#
#     def __repr__(self):
#         return f"_Op({self.kind!r}, {self.name!r}, args={self.args}, kwargs={self.kwargs})"
#
#
# class DeferredResult:
#     """Platzhalter für Rückgabewerte von Methoden (nur Minimalfunktionalität)."""
#     def __init__(self, desc: str = "<deferred>"):
#         self._desc = desc
#
#     def __repr__(self):
#         return f"<DeferredResult {self._desc}>"
#
#
# class _TargetProxy:
#     """
#     Proxy für fig oder ax. Jede Methode/Attributzuweisung wird aufgezeichnet.
#     Interne Namen müssen mit '_' beginnen, damit sie auf das Proxy-Objekt selbst gesetzt werden.
#     Zusätzlich prüfen wir bereits beim Aufruf/Setzen, ob die Matplotlib-Klassen
#     (Figure / Axes) das Attribut / die Methode kennen und werfen frühzeitig
#     einen AttributeError, falls nicht.
#     """
#     _owner: PlotTemplate
#     _target_name: str
#
#
#
#     def __init__(self, owner: 'PlotTemplate', target_name: str):
#         # owner hält die Listen fig_ops bzw. ax_ops
#         # wir umgehen __setattr__ des Objekts, um Rekursion/Nebenwirkungen zu vermeiden
#         object.__setattr__(self, "_owner", owner)
#         object.__setattr__(self, "_target_name", target_name)  # 'fig' oder 'ax'
#
#
#
#     def _target_class(self):
#         return Figure if object.__getattribute__(self, "_target_name") == 'fig' else Axes
#
#
#
#     def __getattr__(self, name: str) -> Callable:
#         # Wenn auf eine nicht-existente Methode/Attribut zugegriffen wird, prüfen wir,
#         # ob die jeweilige Matplotlib-Klasse so ein Attribut / eine Methode besitzt.
#         # Wenn nicht -> frühzeitiger AttributeError. Andernfalls zeichnen wir den Aufruf auf.
#         #
#         # Hinweis: Manche Matplotlib-Attribute werden dynamisch auf Instanzebene erzeugt
#         # und sind nicht als Klassenattribute sichtbar. Diese Prüfung ist eine Best-Effort
#         # Prüfung für frühzeitige Fehlererkennung; in sehr seltenen Fällen kann sie false
#         # negatives liefern (Attribut existiert zur Laufzeit trotzdem auf Instanzen).
#         target_cls = self._target_class()
#         # Akzeptiere Zugriffe auf "private" Namen, die mit '_' beginnen
#         if name.startswith("_"):
#             raise AttributeError(name)
#
#         # Prüfe, ob die Matplotlib-Klasse das Attribut besitzt
#         if not hasattr(target_cls, name):
#             # Falls es ein 'callable' sein sollte aber als Property existiert, ebenfalls prüfen
#             # dir()-Fallback
#             if name not in dir(target_cls):
#                 raise AttributeError(
#                     f"'{target_cls.__name__}' hat kein Attribut oder keine Methode '{name}'"
#                 )
#
#         return self._create_call_recording_function_(name)
#
#
#
#     def _create_call_recording_function_(self, name: str) -> callable:
#         def recorder(*args, **kwargs):
#             self._owner._record(self._target_name, _Op('call', name, args, kwargs))
#             # Wir geben ein DeferredResult zurück — kein echtes Artist-Objekt.
#             return DeferredResult(f"{self._target_name}.{name}()")
#
#         return recorder
#
#
#
#     def __setattr__(self, name: str, value: Any):
#         # interne Attribute direkt setzen
#         if name.startswith("_"):
#             object.__setattr__(self, name, value)
#             return
#
#         # Prüfe, ob die Matplotlib-Klasse dieses Attribut kennt / besitzen sollte
#         target_cls = self._target_class()
#         if not hasattr(target_cls, name):
#             if name not in dir(target_cls):
#                 raise AttributeError(
#                     f"'{target_cls.__name__}' hat kein Attribut '{name}', daher kann es hier nicht gesetzt werden."
#                 )
#
#         # Aufzeichnung der Attributzuweisung
#         self._owner._record(self._target_name, _Op('setattr', name, (value,), {}))
#
#
#
# class FigTemplate(_TargetProxy):
#     def show(self, *args, **kwargs):
#         self._owner._show_(*args, **kwargs)
#
#
#
#     def savefig(self, *args, **kwargs):
#         self._owner._savefig_(*args, **kwargs)
#
#
#
#     # Figure-seitige Komfortmethoden
#     def suptitle(self, *args, **kwargs):
#         """Delegiert an fig.suptitle(...)."""
#         op = self.__getattr__("suptitle")
#         return op(*args, **kwargs)
#
#
#
# class AxTemplate(_TargetProxy):
#     # Einige häufig verwendete Axes-Methoden als "feste" Methoden, damit IDEs sie sehen.
#     # Du kannst hier weitere Methoden hinzufügen, falls gewünscht.
#     def plot(self, *args, **kwargs):
#         """Delegiert an ax.plot(...)."""
#         op = self.__getattr__("plot")
#         return op(*args, **kwargs)
#
#     def scatter(self, *args, **kwargs):
#         """Delegiert an ax.scatter(...)."""
#         op = self.__getattr__("scatter")
#         return op(*args, **kwargs)
#
#     def set_title(self, *args, **kwargs):
#         """Delegiert an ax.set_title(...)."""
#         op = self.__getattr__("set_title")
#         return op(*args, **kwargs)
#
#     def set_xlabel(self, *args, **kwargs):
#         """Delegiert an ax.set_xlabel(...)."""
#         op = self.__getattr__("set_xlabel")
#         return op(*args, **kwargs)
#
#     def set_ylabel(self, *args, **kwargs):
#         """Delegiert an ax.set_ylabel(...)."""
#         op = self.__getattr__("set_ylabel")
#         return op(*args, **kwargs)
#
#     def legend(self, *args, **kwargs):
#         """Delegiert an ax.legend(...)."""
#         op = self.__getattr__("set_ylabel")
#         return op(*args, **kwargs)
#
#
#
# class PlotTemplate:
#     """
#     Hauptklasse. Verwende so:
#         wrapper = DeferredSubplot(figsize=(6,4))
#         wrapper.ax.plot([1,2,3], [1,4,9])
#         wrapper.ax.set_title("Deferred")
#         wrapper.fig.suptitle("My plot")
#         wrapper.savefig("out.png")
#
#     Zusätzliche Komfort-Methoden (z.B. plot, scatter, set_title) sind als
#     direkte Methoden der Klasse implementiert, damit IDEs diese beim Autocomplete
#     vorschlagen. Diese Methoden delegieren intern an die entsprechenden Proxies.
#     """
#
#     fig: FigTemplate
#     ax: AxTemplate
#
#     _subplots_args_: tuple[Any, ...]
#     _subplots_kwargs_: dict[str, Any]
#     _fig_operations_: list[_Op]
#     _ax_operations_: list[_Op]
#     _replayed_real_: tuple[Figure, Axes] | None
#
#
#     @staticmethod
#     def create(*subplots_args, **subplots_kwargs) -> tuple[FigTemplate, AxTemplate]:
#         template: PlotTemplate = PlotTemplate(*subplots_args, **subplots_kwargs)
#
#         return template.fig, template.ax
#
#
#
#     def __init__(self, *subplots_args, **subplots_kwargs):
#         """
#         subplots_args/kwargs werden später an plt.subplots(...) gegeben,
#         wenn das Replay ausgeführt wird. Du kannst hier z. B. figsize=(6,4) setzen.
#         """
#         self._subplots_args_ = subplots_args
#         self._subplots_kwargs_ = subplots_kwargs
#
#         # Listen der aufgezeichneten Operationen
#         self._fig_operations_: List[_Op] = []
#         self._ax_operations_: List[_Op] = []
#
#         # Proxies, die der Benutzer benutzt
#         self.fig = FigTemplate(self, 'fig')
#         self.ax = AxTemplate(self, 'ax')
#
#         # Flag ob bereits ein reales fig/ax erzeugt wurde (bei replay)
#         self._replayed_real_ = None
#
#
#
#     def replay(self):
#         """
#         Erzeuge ein echtes fig, ax (plt.subplots mit den beim Konstruktor
#         gegebenen Args/Kwargs) und wende alle aufgezeichneten Operationen an.
#         Liefert das tatsächliche (fig, ax)-Tupel zurück.
#         """
#         fig, ax = plt.subplots(*self._subplots_args_, **self._subplots_kwargs_)
#         # Wende fig-Operationen an
#         for op in self._fig_operations_:
#             if op.kind == 'call':
#                 func = getattr(fig, op.name, None)
#                 if func is None:
#                     raise AttributeError(f"Figure hat kein Attribut {op.name!r}")
#                 func(*op.args, **op.kwargs)
#             elif op.kind == 'setattr':
#                 setattr(fig, op.name, op.args[0])
#         # Wende ax-Operationen an
#         for op in self._ax_operations_:
#             if op.kind == 'call':
#                 func = getattr(ax, op.name, None)
#                 if func is None:
#                     raise AttributeError(f"Axes hat kein Attribut {op.name!r}")
#                 func(*op.args, **op.kwargs)
#             elif op.kind == 'setattr':
#                 setattr(ax, op.name, op.args[0])
#         # speichere Referenz
#         self._replayed_real_ = (fig, ax)
#         return fig, ax
#
#
#
#     def _show_(self, *args, **kwargs):
#         """
#         Recreate and show. Gibt das reale (fig, ax) zurück.
#         Zusätzliche args/kwargs werden an plt.show() weitergereicht.
#         """
#         fig, ax = self.replay()
#         plt.show(*args, **kwargs)
#         return fig, ax
#
#
#
#     def _savefig_(self, *args, **kwargs):
#         """
#         Replay und savefig auf dem wiederhergestellten figure ausführen.
#         Beispiel: wrapper.savefig("out.png", dpi=150)
#         """
#         fig, ax = self.replay()
#         fig.savefig(*args, **kwargs)
#         return fig, ax
#
#
#
#     def clear_recordings(self):
#         """Löscht alle aufgezeichneten Operationen (nicht das bereits gerenderte fig/ax)."""
#         self._fig_operations_.clear()
#         self._ax_operations_.clear()
#         self._replayed_real_ = None
#
#
#
#     def get_recorded_ops(self):
#         """Hilfsfunktion zur Inspektion im Debugging."""
#         return {'fig': list(self._fig_operations_), 'ax': list(self._ax_operations_)}
#
#
#
#     def _record(self, target: str, op: _Op):
#         if target == 'fig':
#             self._fig_operations_.append(op)
#         elif target == 'ax':
#             self._ax_operations_.append(op)
#         else:
#             raise ValueError("target must be 'fig' or 'ax'")


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
            # interne Variablen vom Wrapper selber
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
                value, = rest
                setattr(internal_obj, name, value)



    @abstractmethod
    def _get_internal_object_(self) -> plt.Figure | plt.Axes:
        pass



    # def __repr__(self) -> str:
    #     return f"{self.__class__.__name__}(log={repr(self._log_)})"
    #
    # def __str__(self) -> str:
    #     return f"{self.__class__.__name__}(log={str(self._log_)})"



class FigureWrapper(WrapperBase):
    # Figure-seitige Komfortmethoden
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
    # Einige häufig verwendete Axes-Methoden als "feste" Methoden, damit IDEs sie sehen.
    # Du kannst hier weitere Methoden hinzufügen, falls gewünscht.
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
    _internal_figure_axes_dict_: dict[object, tuple[plt.Figure, plt.Axes]] = weakref.WeakKeyDictionary()


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


