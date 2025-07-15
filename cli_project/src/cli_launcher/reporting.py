from pipeline_entities.output.pipeline_component_execution_report import PipelineComponentExecutionReport
import jax.numpy as jnp
from typing import Any
from pipeline_entities.pipeline_component_instantiation_info.pipeline_component_instantiation_info import PipelineComponentInstantiationInfo
from pipeline_entities.data_transfer.pipeline_data import PipelineData

def format_report(report: PipelineComponentExecutionReport) -> str:
    info = report.component_instantiation_info
    data = report.component_output

    comp_name = info.component_name
    comp_id = info.component.component_id
    title = f"{comp_name} {comp_id}"

    # timings
    init_t = report.component_init_time
    exec_t = report.component_execution_time
    timing_lines = []
    if init_t is not None:
        timing_lines.append(f"Initialization time: {init_t * 1000:.3f} ms")
    if exec_t is not None:
        timing_lines.append(f"Execution time: {exec_t * 1000:.3f} ms")

    rows: list[tuple[str, list[str]]] = []  # list of tuples: (field_name, list of summary lines)
    if data is not None:
        for field_name, field_def in data.__dataclass_fields__.items():
            val = getattr(data, field_name)
            if val is None:
                continue
            if isinstance(val, jnp.ndarray):
                summary_str = str(val)
            elif isinstance(val, (dict, list, tuple)):
                summary_str = repr(val)
            else:
                summary_str = str(val)
            summary_lines = summary_str.splitlines()
            rows.append((field_name, summary_lines))
    else:
        rows.append(("<no output>", [" "]))

    # column widths
    left_header = "Field"
    right_header = "Value"
    left_width = max(len(left_header), *(len(r[0]) for r in rows))
    right_width = max(len(right_header), *(len(line) for r in rows for line in r[1]))

    # table borders
    sep_line = f"+{'-' * (left_width + 2)}+{'-' * (right_width + 2)}+"
    header_fmt = f"| {{:{left_width}}} | {{:{right_width}}} |"

    # full width for title underline
    table_width = len(sep_line)
    underline = '=' * table_width
    centered_title = title.center(table_width)

    lines: list[str] = [underline, centered_title, underline, ""]

    for l in timing_lines:
        lines.append(l)
    lines.append("") # add blank line for better readability

    # Start of Table
    lines.append("Pipeline Data:")
    lines.append(sep_line)
    lines.append(header_fmt.format(left_header, right_header))
    lines.append(sep_line)

    # data rows
    for field_name, summary_lines in rows:
        for i, txt in enumerate(summary_lines):
            if i == 0:
                lines.append(header_fmt.format(field_name, txt))
            else:
                lines.append(header_fmt.format('', txt))
        lines.append(sep_line)

    return '\n'.join(lines)

def format_all_reports(reports: list[PipelineComponentExecutionReport]) -> str:
    return "\n\n\n".join(format_report(r) for r in reports)