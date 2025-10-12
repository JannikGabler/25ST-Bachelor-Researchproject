from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from file_handling.result_persistence.save_policy import SavePolicy

class Saver(ABC):
    """
    Abstract base for savers.
    """
    kind: str # e.g., "plot", "json", "jnp_array", ...

    @abstractmethod
    def save(self, artifact: Any, run_dir: Path, policy: SavePolicy) -> Path:
        """Persist the artifact and return the primary file path."""
        raise NotImplementedError
    
_SAVER_REGISTRY: dict[str, Saver] = {}

def register_saver(saver: Saver) -> None: 
    if not saver.kind: 
        raise ValueError(f"saver.kind must be a non-empty string. Got: {saver.kind}") 
    _SAVER_REGISTRY[saver.kind] = saver

def get_saver(kind: str) -> Saver:
    try:
        return _SAVER_REGISTRY[kind]
    except KeyError as e:
        raise ValueError(f"No saver registered for artifact kind '{kind}") from e
