from __future__ import annotations
from pathlib import Path
from typing import Any

from data_classes.plot_template.plot_template import PlotTemplate, FigureWrapper
from file_handling.result_persistence.save_policy import SavePolicy
from file_handling.result_persistence.savers.base import Saver, register_saver
from file_handling.result_persistence.utils import slugify


class PlotSaver(Saver):
    """Saves PlotTemplate-based figures (FigureWrapper) to disk."""

    kind = "plot"

    def _get_figure_wrapper(self, plot_template: PlotTemplate) -> FigureWrapper:
        """
        Accept either:
          - PlotTemplate.fig as a FigureWrapper (your case), or
          - a callable that returns a FigureWrapper (legacy)
          - an object with a .savefig method (fallback)
        """
        fig_attr: Any = getattr(plot_template, "fig", None)
        if fig_attr is None:
            raise AttributeError("PlotTemplate has no 'fig' attribute.")

        wrapper = (
            fig_attr() if callable(fig_attr) else fig_attr
        )  # support legacy callable style TODO should be safe to remove

        if not hasattr(wrapper, "savefig"):
            raise TypeError(
                "PlotTemplate.fig did not provide an object with a 'savefig' method."
            )
        return wrapper  # type: ignore[return-value]

    def _base_name(self, plot_template: PlotTemplate) -> str:
        # Try common fields; otherwise use a generic name
        raw = (
            getattr(plot_template, "name", None)
            or getattr(plot_template, "title", None)
            or "plot"
        )
        return slugify(str(raw))

    def _unique_stem(self, directory: Path, stem: str) -> str:
        """Avoid collisions for identical names; keep deterministic and simple."""
        if not any(directory.glob(f"{stem}.*")):
            return stem
        i = 1
        while any(directory.glob(f"{stem}-{i}.*")):
            i += 1
        return f"{stem}-{i}"

    def save(
        self, plot_template: PlotTemplate, run_dir: Path, policy: SavePolicy
    ) -> Path:
        # Place plots in a 'plots' subfolder of the provided run_dir (component dir or run root)
        plot_dir = run_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        stem = self._base_name(plot_template)
        stem = self._unique_stem(plot_dir, stem)

        # Defaults #TODO move into SavePolicy
        exts: list[str] = list(policy.plot_formats)
        dpi = 150
        transparent = False

        fig_wrapper: FigureWrapper = self._get_figure_wrapper(plot_template)

        primary_path: Path | None = None
        for ext in exts:
            out = plot_dir / f"{stem}.{ext}"
            try:
                fig_wrapper.savefig(
                    out,
                    dpi=dpi,
                    format=ext,
                    bbox_inches="tight",
                    transparent=transparent,
                )
                if primary_path is None:
                    primary_path = out
            except Exception as e:
                print(f"⚠️ Warning: Could not save plot {stem} as .{ext}: {e}")
                continue

        if primary_path is None:
            raise RuntimeError(f"Could not save plot '{stem}' with formats {exts}.")

        return primary_path


register_saver(PlotSaver())
