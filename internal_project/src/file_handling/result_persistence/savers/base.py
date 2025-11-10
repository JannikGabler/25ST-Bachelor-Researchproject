from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from file_handling.result_persistence.save_policy import SavePolicy


class Saver(ABC):
    """
    Abstract base for savers.
    """


    kind: str


    @abstractmethod
    def save(self, artifact: Any, run_dir: Path, policy: SavePolicy) -> Path:
        """
        Persist the given artifact according to the save policy and return the primary output file path.

        Args:
            artifact (Any): The object or data to be saved (e.g., reports, plots, etc.).
            run_dir (Path): Target run directory where the artifact will be stored.
            policy (SavePolicy): Policy object defining how the artifact should be saved (e.g., format, indentation, rotation).

        Returns:
            Path: Path to the main saved file.
        """

        raise NotImplementedError


_SAVER_REGISTRY: dict[str, Saver] = {}


def register_saver(saver: Saver) -> None:
    """
    Register a Saver instance under its 'kind' identifier for later retrieval.

    Args:
        saver (Saver): Saver instance to register. Must define a non-empty 'kind' attribute.

    Returns:
        None
    """

    if not saver.kind:
        raise ValueError(f"saver.kind must be a non-empty string. Got: {saver.kind}")
    _SAVER_REGISTRY[saver.kind] = saver


def get_saver(kind: str) -> Saver:
    """
    Retrieve a registered Saver instance by its kind identifier.

    Args:
        kind (str): The saver kind string (e.g., 'reports', 'plot').

    Returns:
        Saver: The corresponding registered Saver instance.

    Raises:
        ValueError: If no saver is registered under the given kind.
    """

    try:
        return _SAVER_REGISTRY[kind]
    except KeyError as e:
        raise ValueError(f"No saver registered for artifact kind '{kind}") from e
