from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

@dataclass(frozen=True)
class SavePolicy:
    """
    Persistence policy passed to the store.
    - mode:
        * "soft-state": write to <output_root>/runs/latest (overwritten next run)
        * "snapshot":   write to timestamped directory (immutable)
    - plot_formats:    order matters; first is preferred (e.g., ["png", "pdf"])
    - json_indent:     pretty print JSON
    - keep_soft_state_n: if >1, keep rotating latest directories latest, latest-1, ...
    """
    mode: Literal["soft-state", "snapshot"] = "soft-state"
    plot_formats: list[str] = field(default_factory=lambda: ["png"])
    json_indent: int = 2
    keep_soft_state_n: int = 1

# TODO default values into internal_project/src/constants/internal_logic_constants.py
