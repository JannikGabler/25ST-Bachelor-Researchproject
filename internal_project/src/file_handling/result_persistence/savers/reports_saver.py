from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Iterable

from file_handling.result_persistence.save_policy import SavePolicy
from file_handling.result_persistence.savers.base import Saver, register_saver

from pipeline_entities.pipeline.component_entities.component_info.dataclasses.pipeline_component_info import (
    PipelineComponentInfo,
)
from pipeline_entities.pipeline_execution.dataclasses.pipeline_component_instantiation_info import (
    PipelineComponentInstantiationInfo,
)
from pipeline_entities.pipeline_execution.output.pipeline_component_execution_report import (
    PipelineComponentExecutionReport,
)

from file_handling.result_persistence.utils import type_repr, to_jsonable, prune_trivial


def _serialize_report(rep: PipelineComponentExecutionReport) -> dict[str, Any]:
    """
    Convert a PipelineComponentExecutionReport into a JSON-serializable dict.
    Keeps it lean and stable; avoids dumping large PipelineData contents.
    """
    comp_inst_info: PipelineComponentInstantiationInfo = (
        rep.component_instantiation_info
    )
    comp_info: PipelineComponentInfo = comp_inst_info.component

    comp_name = getattr(comp_inst_info, "component_name", None) or getattr(
        comp_info, "name", None
    )
    comp_id = getattr(comp_info, "component_id", None)
    comp_type = type_repr(getattr(comp_info, "component_type", None))
    comp_class = type_repr(getattr(comp_info, "component_class", None))

    pdata = rep.component_output
    plots = getattr(pdata, "plots", None) if pdata is not None else None
    plots_count = len(plots) if plots else 0

    pd_json = prune_trivial(to_jsonable(pdata)) if pdata is not None else None

    return {
        "component": {
            "id": comp_id,
            "name": comp_name,
            "type": comp_type,
            "class": comp_class,
            # could add more metadata (version, tags, etc.)
        },
        "timing": {
            "init_time": rep.component_init_time,
            "avg_exec_time": rep.average_component_execution_time,
            "std_exec_time": rep.standard_deviation_component_execution_time,
        },
        "outputs": {"plots_count": plots_count, "pipeline_data": pd_json},
    }


class ReportsSaver(Saver):
    """Saves a list of PipelineComponentExecutionReport to a single JSON."""

    kind = "reports"

    def save(
        self,
        artifact: (
            Iterable[PipelineComponentExecutionReport]
            | PipelineComponentExecutionReport
        ),
        run_dir: Path,
        policy: SavePolicy,
    ) -> Path:
        reports_dir = run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Normalize to list
        if isinstance(artifact, PipelineComponentExecutionReport):
            payload = [_serialize_report(artifact)]
        else:
            payload = [_serialize_report(r) for r in artifact]

        out = reports_dir / "reports.json"
        out.write_text(json.dumps(payload, indent=policy.json_indent), encoding="utf-8")
        return out


register_saver(ReportsSaver())
