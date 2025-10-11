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
    """
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text.strip())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or "artifact"

def type_repr(t) -> str | None:
    if t is None:
        return None
    if isinstance(t, str):
        return t
    # Handle typing constructs nicely (e.g., list[int], dict[str, Any], X | Y)
    try:
        origin = get_origin(t)
        if origin:
            args = ", ".join((type_repr(a) or "None") for a in get_args(t))
            return f"{origin.__module__}.{origin.__qualname__}[{args}]"  # e.g. builtins.list[int]
        # Regular classes/types
        mod = getattr(t, "__module__", None) or "<unknown>"
        qn = getattr(t, "__qualname__", None) or getattr(t, "__name__", repr(t))
        return f"{mod}.{qn}"
    except Exception:
        # Last-resort readable fallback
        return repr(t)
    
def callable_repr(obj: Any) -> str:
    # For functions, methods, callables—give a readable id
    try:
        if inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isbuiltin(obj):
            mod = getattr(obj, "__module__", "<unknown>")
            qn = getattr(obj, "__qualname__", getattr(obj, "__name__", repr(obj)))
            return f"{mod}.{qn}"
        # Callable instance or object with __call__
        cls = obj.__class__
        return f"{cls.__module__}.{cls.__qualname__}(callable)"
    except Exception:
        return repr(obj)

def classlike_repr(obj: Any) -> str:
    try:
        return type_repr(obj)
    except Exception:
        return repr(obj)

def ndarray_like_to_list(arr):
    try:
        return arr.tolist()
    except Exception:
        # Last resort: convert to bytes length / shape summary to avoid crashes
        shape = getattr(arr, "shape", None)
        dtype = getattr(arr, "dtype", None)
        return {"__array__": "unserializable", "shape": tuple(shape) if shape else None, "dtype": str(dtype)}

def dtype_to_str(dt) -> str:
    try:
        return str(dt)
    except Exception:
        return repr(dt)

def to_jsonable(obj: Any, _seen: set[int] | None = None) -> Any:
    """
    Convert arbitrary Python/JAX/NumPy / dataclass objects to JSON-friendly structures.
    """
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return "<recursion>"
    _seen.add(oid)

    # Primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # NumPy / JAX arrays
    if jnp is not None and isinstance(obj, (jnp.ndarray,)):  # type: ignore[arg-type]
        return ndarray_like_to_list(obj)
    if np is not None and isinstance(obj, (np.ndarray,)):    # type: ignore[arg-type]
        return ndarray_like_to_list(obj)

    # NumPy / JAX scalars
    if np is not None and isinstance(obj, (np.generic,)):    # type: ignore[arg-type]
        return obj.item()
    if jnp is not None and hasattr(jnp, "scalar_types") and isinstance(obj, tuple(getattr(jnp, "scalar_types", ()))):  # defensive
        try:
            return obj.item()
        except Exception:
            pass

    # dtypes
    if np is not None and isinstance(obj, (np.dtype,)):      # type: ignore[arg-type]
        return dtype_to_str(obj)

    # Containers
    if isinstance(obj, dict):
        return {str(to_jsonable(k, _seen)): to_jsonable(v, _seen) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(x, _seen) for x in obj]

    # Dataclasses
    if is_dataclass(obj):
        return {k: to_jsonable(v, _seen) for k, v in asdict(obj).items()}

    # Types / classes
    if isinstance(obj, type):
        return classlike_repr(obj)

    # Callables (functions, compiled functions, etc.)
    if callable(obj):
        return callable_repr(obj)

    # Objects with a helpful __json__ or to_dict
    for attr in ("__json__", "to_json", "to_dict", "as_dict"):
        if hasattr(obj, attr) and callable(getattr(obj, attr)):
            try:
                return to_jsonable(getattr(obj, attr)(), _seen)
            except Exception:
                pass

    # Fallback: class + shallow public attrs (small, safe summary)
    try:
        cls = obj.__class__
        public = {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        # Guard against huge data: recurse but it will handle arrays/dicts, etc.
        return {
            "__class__": f"{cls.__module__}.{cls.__qualname__}",
            **{k: to_jsonable(v, _seen) for k, v in public.items()}
        }
    except Exception:
        # Last resort: readable string
        return repr(obj)
