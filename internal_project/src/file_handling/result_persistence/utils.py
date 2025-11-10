from __future__ import annotations

import inspect
import numpy as np
import jax.numpy as jnp

from dataclasses import is_dataclass, asdict
from typing import Any, get_origin, get_args


def slugify(text: str) -> str:
    """
    Convert an arbitrary string into a filesystem-safe slug.
    - Keeps only alphanumeric characters, dashes (`-`), and underscores (`_`).
    - Replaces all other characters with underscores.
    - Collapses consecutive underscores into a single underscore.
    - Strips leading/trailing whitespace before processing.
    - Returns "artifact" if the result would otherwise be empty.

    Args:
        text (str): The input string to be normalized.

    Returns:
        str: A safe slug suitable for filenames/keys.
    """

    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text.strip())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or "artifact"


def type_repr(t) -> str | None:
    """
    Produce a stable, readable string for a Python type or typing construct.
    Handles:
      - `None` -> None
      - Raw strings (returned as-is)
      - Standard classes/types -> "module.qualname"
      - Typing constructs (e.g., list[int], dict[str, Any]) -> "module.qualname[arg1, arg2]"

    Args:
        t (Any): A type object, typing annotation, or string.

    Returns:
        str | None: Readable identifier or None if input is None.
    """

    if t is None:
        return None
    if isinstance(t, str):
        return t

    try:
        origin = get_origin(t)
        if origin:
            args = ", ".join((type_repr(a) or "None") for a in get_args(t))
            return f"{origin.__module__}.{origin.__qualname__}[{args}]"  # e.g. builtins.list[int]
        mod = getattr(t, "__module__", None) or "<unknown>"
        qn = getattr(t, "__qualname__", None) or getattr(t, "__name__", repr(t))
        return f"{mod}.{qn}"
    except Exception:
        return repr(t)


def callable_repr(obj: Any) -> str:
    """
    Produce a readable identifier for callables (functions, methods, builtins, or callable instances).
    For functions/methods/builtins: "module.qualname".
    For callable instances: "module.ClassName(callable)".

    Args:
        obj (Any): Callable object or any object (falls back to repr on error).

    Returns:
        str: Readable identifier string.
    """

    try:
        if inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isbuiltin(obj):
            mod = getattr(obj, "__module__", "<unknown>")
            qn = getattr(obj, "__qualname__", getattr(obj, "__name__", repr(obj)))
            return f"{mod}.{qn}"
        cls = obj.__class__
        return f"{cls.__module__}.{cls.__qualname__}(callable)"
    except Exception:
        return repr(obj)


def classlike_repr(obj: Any) -> str:
    """
    Wrapper around `type_repr` that tolerates failures and returns `repr(obj)` as fallback.

    Args:
        obj (Any) : A class/type or typing construct.

    Returns:
        str: Readable class/type representation.
    """

    try:
        return type_repr(obj)
    except Exception:
        return repr(obj)


def ndarray_like_to_list(arr):
    """
    Convert NumPy/JAX array-likes to Python lists. If `.tolist()` fails, return a small summary dict containing shape
    and dtype instead of raising.

    Args:
        arr (Any) : NumPy/JAX ndarray-like object.

    Returns:
        list | dict: A list for successful conversions; otherwise a summary dict.
    """

    try:
        return arr.tolist()
    except Exception:
        shape = getattr(arr, "shape", None)
        dtype = getattr(arr, "dtype", None)
        return {
            "__array__": "unserializable",
            "shape": tuple(shape) if shape else None,
            "dtype": str(dtype),
        }


def dtype_to_str(dt) -> str:
    """
    Convert a dtype-like object to a robust string representation.

    Args:
    dt (Any) : A NumPy dtype or similar.

    Returns:
        str: String form of the dtype; falls back to repr on error.
    """

    try:
        return str(dt)
    except Exception:
        return repr(dt)


def to_jsonable(obj: Any, _seen: set[int] | None = None) -> Any:
    """
    Convert arbitrary Python/JAX/NumPy/dataclass objects to JSON-friendly structures.

    Rules of conversion:
      - Primitives (None, bool, int, float, str) -> unchanged.
      - NumPy/JAX arrays -> lists (via `.tolist()`), with a safe fallback summary.
      - NumPy scalars -> `.item()`.
      - dtypes -> strings.
      - dict/list/tuple/set/frozenset -> recursively converted.
      - dataclasses -> `asdict` then recursively converted.
      - `type` objects -> `classlike_repr`.
      - Callables -> `callable_repr`.
      - Objects with `__json__`, `to_json`, `to_dict`, `as_dict` -> call and recurse.
      - Fallback -> dict of public attributes with `"__class__"` marker.
      - Cycles are detected and represented as the string "<recursion>".

    Args:
        obj (Any) : Object to serialize.
        _seen (set[int] | None, optional) : Internal set of visited object IDs to prevent infinite recursion.

    Returns:
        Any: A structure composed only of JSON-serializable primitives, lists, and dicts.
    """

    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return "<recursion>"
    _seen.add(oid)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if jnp is not None and isinstance(obj, (jnp.ndarray,)):
        return ndarray_like_to_list(obj)
    if np is not None and isinstance(obj, (np.ndarray,)):
        return ndarray_like_to_list(obj)
    if np is not None and isinstance(obj, (np.generic,)):
        return obj.item()
    if jnp is not None and hasattr(jnp, "scalar_types") and isinstance(obj, tuple(getattr(jnp, "scalar_types", ()))):
        try:
            return obj.item()
        except Exception:
            pass
    if np is not None and isinstance(obj, (np.dtype,)):
        return dtype_to_str(obj)
    if isinstance(obj, dict):
        return {str(to_jsonable(k, _seen)): to_jsonable(v, _seen) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(x, _seen) for x in obj]
    if is_dataclass(obj):
        return {k: to_jsonable(v, _seen) for k, v in asdict(obj).items()}
    if isinstance(obj, type):
        return classlike_repr(obj)
    if callable(obj):
        return callable_repr(obj)
    for attr in ("__json__", "to_json", "to_dict", "as_dict"):
        if hasattr(obj, attr) and callable(getattr(obj, attr)):
            try:
                return to_jsonable(getattr(obj, attr)(), _seen)
            except Exception:
                pass
    try:
        cls = obj.__class__
        public = {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        return {"__class__": f"{cls.__module__}.{cls.__qualname__}", **{k: to_jsonable(v, _seen) for k, v in public.items()}}
    except Exception:
        return repr(obj)


def _is_trivial(v: Any) -> bool:
    if v is None:
        return True
    if v == "<recursion>":
        return True
    if isinstance(v, (list, tuple, set, frozenset)) and len(v) == 0:
        return True
    if isinstance(v, dict) and len(v) == 0:
        return True
    return False


def prune_trivial(obj: Any) -> Any:
    """
    Recursively remove trivial entries from a JSON-like structure.

    Trivial entries are:
    - None
    - the string "<recursion>"
    - empty lists/tuples/sets/frozensets
    - empty dicts

    Notes
    - Dict keys whose values become trivial are dropped.
    - Sequences are returned as lists with trivial items removed.
    - Non-container values are returned unchanged.
    """

    if isinstance(obj, dict):
        pruned = {k: prune_trivial(v) for k, v in obj.items()}
        return {k: v for k, v in pruned.items() if not _is_trivial(v)}
    if isinstance(obj, (list, tuple, set, frozenset)):
        pruned_seq = [prune_trivial(v) for v in obj]
        pruned_seq = [v for v in pruned_seq if not _is_trivial(v)]
        return pruned_seq if not isinstance(obj, tuple) else tuple(pruned_seq)
    return obj
