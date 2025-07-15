from pipeline_entities.output.pipeline_component_execution_report import PipelineComponentExecutionReport
import jax.numpy as jnp
from typing import Any
from pipeline_entities.pipeline_component_instantiation_info.pipeline_component_instantiation_info import PipelineComponentInstantiationInfo
from pipeline_entities.data_transfer.pipeline_data import PipelineData

def format_report(report: PipelineComponentExecutionReport) -> str:
    info = report.component_instantiation_info
    data = report.component_output

    # Header: component name and id
    comp_name = info.component_name
    comp_id = info.component.component_id
    header = f"Component {comp_name} (ID: '{comp_id}')"
    underline = "=" * len(header)

    lines = [underline, header, underline]

    # Timings
    init_t = report.component_init_time
    exec_t = report.component_execution_time
    if init_t is not None:
        lines.append(f"Init time:      {init_t * 1000:.3f} ms")
    if exec_t is not None:
        lines.append(f"Exec time:      {exec_t * 1000:.3f} ms")

    # Output summary
    lines.append("\nOutput Summary:")
    if data is None:
        lines.append("  <no output>")
    else:
        # Inspect each field of PipelineData
        # Only include non-None entries
        for field_name in data.__dataclass_fields__:
            val = getattr(data, field_name)
            if val is None:
                continue
            if isinstance(val, jnp.ndarray):
                # summarize JAX array
                summary = f"array shape={val.shape}, dtype={val.dtype}"
            elif isinstance(val, (dict, list, tuple)):
                summary = repr(val)
            else:
                summary = str(val)
            lines.append(f"  {field_name}: {summary}")
    lines.append(underline)
    return "\n".join(lines)


def format_all_reports(reports: list[PipelineComponentExecutionReport]) -> str:
    return "\n\n".join(format_report(r) for r in reports)