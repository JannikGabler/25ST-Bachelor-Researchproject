from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Iterable

from data_classes.plot_template.plot_template import PlotTemplate
from file_handling.result_persistence.save_policy import SavePolicy
from file_handling.result_persistence.savers.base import get_saver


from constants.internal_logic_constants import FilesystemResultStoreConstants

from file_handling.result_persistence.savers import plot_saver
from file_handling.result_persistence.savers import reports_saver

from pipeline_entities.pipeline_execution.output.pipeline_component_execution_report import PipelineComponentExecutionReport

from pipeline_entities.pipeline.component_entities.component_info.dataclasses.pipeline_component_info import PipelineComponentInfo


class FilesystemResultStore:
    """
    Minimal, purpose-built store for this project.
    Responsibilities:
      1) Create a run directory according to SavePolicy.
      2) Save execution reports -> JSON (single file).
      3) Save plots contained in PipelineData.plots, grouped per component.
    Directory layout (example):
      runs/
        latest/ or 2025-10-08_14-37-55/
          reports/
            reports.json
          components/
            <component_id>/
              plots/
                plot.png, plot.pdf, ...
    """


    def __init__(self, output_root: Path):
        """
        Args:
            output_root: Root directory under which the 'runs/' folder will be created.
        """

        self.output_root = Path(output_root)
        self.runs_root = self.output_root / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)


    ################################
    ### Run directory management ###
    ################################
    def prepare_run_dir(self, policy: SavePolicy) -> Path:
        """
        Create and return the run directory based on the policy.
        soft-state -> runs/latest (with optional rotation);
        snapshot   -> runs/<timestamp>.

        Args:
            policy (SavePolicy): Defines how and where run data should be stored.

        Returns:
            Path: Path to the newly created run directory.
        """

        if policy.mode == "soft-state":
            run_dir = self.runs_root / "latest"
            if run_dir.exists() and policy.keep_soft_state_n > 1:
                self._rotate_soft_state(run_dir, policy.keep_soft_state_n)
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
        elif policy.mode == "snapshot":
            run_dir = self._new_timestamped_dir()
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(f"Unknown SavePolicy.mode: {policy.mode}")
        return run_dir


    def promote_latest_to_snapshot(self) -> Path:
        """
        Copy runs/latest into a new timestamped snapshot directory and return the destination path.

        Returns:
            Path: Path to the newly created snapshot directory.
        """

        src = self.runs_root / "latest"
        if not src.exists():
            raise FileNotFoundError("No 'latest' directory to promote.")
        dst = self._new_timestamped_dir()
        shutil.copytree(src, dst)
        return dst


    ######################
    ### Public methods ###
    ######################

    def save_run(self, reports: Iterable[PipelineComponentExecutionReport], policy: SavePolicy | None) -> Path:
        """
        Orchestrates saving both execution reports and plots for a single pipeline run. Returns the run directory.

        Args:
            reports (Iterable[PipelineComponentExecutionReport]: Execution reports for all pipeline components.
            policy (SavePolicy | None): Storage policy; if None, defaults to FilesystemResultStoreConstants.POLICY.

        Returns:
            Path: Path to the created run directory.
        """

        if policy is None:
            policy = FilesystemResultStoreConstants.POLICY

        run_dir = self.prepare_run_dir(policy)

        reports = list(reports)
        self.save_reports(reports, run_dir, policy)
        self.save_plots_from_reports(reports, run_dir, policy)

        return run_dir


    def save_reports(self, reports: Iterable[PipelineComponentExecutionReport], run_dir: Path, policy: SavePolicy) -> Path:
        """
        Save all component execution reports into a single JSON file.

        Args:
            reports (Iterable[PipelineComponentExecutionReport]: Reports to serialize and save.
            run_dir (Path): Target run directory (e.g. runs/latest or runs/<timestamp>).
            policy (SavePolicy): Storage policy defining formatting and structure.

        Returns:
            Path: Path to the created JSON file (e.g. <run>/reports/reports.json).
        """

        saver = get_saver("reports")
        return saver.save(list(reports), run_dir, policy)


    def save_plots_from_reports(self, reports: Iterable[PipelineComponentExecutionReport], run_dir: Path, policy: SavePolicy) -> None:
        """
        Extract plots from each report's PipelineData and save them, grouped by component_id: runs/<run>/components/<id>/plots/*

        Args:
            reports (Iterable[PipelineComponentExecutionReport]: Reports containing plots to extract and save.
            run_dir (Path): Target run directory.
            policy (SavePolicy): Storage policy defining output formats and behavior.

        Returns:
            None
        """

        plot_saver = get_saver("plot")

        for report in reports:
            comp_info: PipelineComponentInfo = (report.component_instantiation_info.component)
            comp_id: str = getattr(comp_info, "component_id", "unknown_component")

            pdata = report.component_output
            if pdata is None:
                continue
            plots: list[PlotTemplate] = getattr(pdata, "plots", None) or []
            if not plots:
                continue

            component_root = run_dir / "components" / comp_id
            component_root.mkdir(parents=True, exist_ok=True)

            for plot in plots:
                try:
                    plot_saver.save(plot, component_root, policy)
                except Exception as e:
                    # one 'bad' plot doesn't stop execution
                    print(f"⚠️ Could not save plot for component {comp_id}: {e}")


    #######################
    ### Private methods ###
    #######################
    def _new_timestamped_dir(self) -> Path:
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        path = self.runs_root / ts
        idx = 1
        while path.exists():
            path = self.runs_root / f"{ts}_{idx}"
            idx += 1
        return path


    @staticmethod
    def _rotate_soft_state(latest_dir: Path, keep_n: int) -> None:
        base = latest_dir.parent
        oldest = base / f"latest-{keep_n-1}"
        if oldest.exists():
            shutil.rmtree(oldest)
        for i in range(keep_n - 2, -1, -1):
            src_name = "latest" if i == 0 else f"latest-{i}"
            src = base / src_name
            if src.exists():
                dst = base / f"latest-{i+1}"
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(str(src), str(dst))
