from pipeline_entities.output.pipeline_component_execution_report import PipelineComponentExecutionReport
import jax.numpy as jnp
from typing import Any
from pipeline_entities.pipeline_component_instantiation_info.pipeline_component_instantiation_info import PipelineComponentInstantiationInfo
from pipeline_entities.data_transfer.pipeline_data import PipelineData

def _title(report: PipelineComponentExecutionReport) -> str:
    """
    Build title string from component name and ID.
    """
    info = report.component_instantiation_info
    return f"{info.component_name} {info.component.component_id}"

def _times(report: PipelineComponentExecutionReport) -> list[str]:
    """
    Extract initialization and execution times in milliseconds.
    """
    lines: list[str] = []
    if report.component_init_time is not None:
        lines.append(f"Initialization time: {report.component_init_time * 1000:.3f} ms")
    if report.component_execution_time is not None:
        lines.append(f"Execution time: {report.component_execution_time * 1000:.3f} ms")
    return lines

def _prepare_rows(data: PipelineData | None) -> list[tuple[str, list[str]]]:
    """
    Convert PipelineData fields into (field_name, [value_lines]).
    Include only non-None fields.
    """
    rows: list[tuple[str, list[str]]] = [] # list of tuples: (field_name, list of summary lines)
    if data is None:
        return [("<no output>", [" "])]
    for field_name, _ in data.__dataclass_fields__.items():
        val = getattr(data, field_name)
        if val is None:
            continue
        if isinstance(val, jnp.ndarray):
            summary = str(val)
        elif isinstance(val, (dict, list, tuple)):
            summary = repr(val)
        else:
            summary = str(val)
        rows.append((field_name, summary.splitlines()))
    return rows

def _compute_column_widths(rows: list[tuple[str, list[str]]], left_header: str, right_header: str) -> tuple[int, int]:
    """
    Determine the widths for the 2 columns.
    """
    left_width = max(len(left_header), *(len(r[0]) for r in rows))
    right_width = max(len(right_header), *(len(line) for _, lines in rows for line in lines))
    return left_width, right_width

def _format_table(rows: list[tuple[str, list[str]]], left_header: str, right_header: str) -> tuple[list[str], int]:
    """
    Render the table as ASCII lines given rows and column widths.
    """
    left_width, right_width = _compute_column_widths(rows, left_header, right_header)
    sep = f"+{'-' * (left_width + 2)}+{'-' * (right_width + 2)}+"
    fmt = f"| {{:{left_width}}} | {{:{right_width}}} |"
    lines: list[str] = [sep, fmt.format(left_header, right_header), sep]
    for name, value_lines in rows:
        for i, line in enumerate(value_lines):
            if i == 0:
                lines.append(fmt.format(name, line))
            else:
                lines.append(fmt.format('', line))
        lines.append(sep)
    return lines, len(sep)

def format_report(report: PipelineComponentExecutionReport) -> str:
    title = _title(report)
    times = _times(report)

    # Table
    left_header, right_header = "Field", "Value"

    rows = _prepare_rows(report.component_output)
    table, width = _format_table(rows, left_header, right_header)
    underline = '=' * width
    centered_title = title.center(width)

    output: list[str] = [underline, centered_title, underline, '']
    if times:
        output.extend(times + [''])
    output.append("Output (PipelineData):")
    output.extend(table)

    return '\n'.join(output) # list[str] => str (one line per string in the list)

def format_all_reports(reports: list[PipelineComponentExecutionReport]) -> str:
    return "\n\n\n".join(format_report(r) for r in reports)