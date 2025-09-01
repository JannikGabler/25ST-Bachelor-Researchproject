import inspect
import jax.numpy as jnp

from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline_execution.output.pipeline_component_execution_report import \
    PipelineComponentExecutionReport


def _info() -> str:
    return "" \
    "#####################################################################################################\n" \
    "#                                                                                                   #\n" \
    "# INFO: Only new/changed outputs are shown in the tables per component. If not specified otherwise, #\n" \
    "# a field that isn't in a component's table has the same value as the previous component.           #\n" \
    "#                                                                                                   #\n" \
    "#####################################################################################################\n"

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
    if report.average_component_execution_time is not None:
        lines.append(f"Execution time: {report.average_component_execution_time * 1000:.3f} ms")
    return lines

def _rows(data: PipelineData | None, previous: list[tuple[str, list[str]]] | None = None) -> list[tuple[str, list[str]]]:
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
        found = False
        if callable(val): # for functions (lambdas), e.g. function_callable
            try:
                summary = inspect.getsource(val).strip()
                found = True
            except (OSError, TypeError):
                pass
        if not found:
            if isinstance(val, jnp.ndarray):
                summary = str(val)
            elif isinstance(val, (dict, list, tuple)):
                summary = repr(val)
            else:
                summary = str(val)
        result_tuple = (field_name, summary.splitlines())
        if (previous is None) or (not result_tuple in previous):
            rows.append(result_tuple)
    return rows

def _table(rows: list[tuple[str, list[str]]], left_header: str, right_header: str) -> tuple[list[str], int]:
    """
    Render the table as ASCII lines given rows and column widths.
    """
    if not rows:
        return (out := ["<Nothing new>"], len(out[0]))

    # Column widths
    left_width  = len(left_header)
    right_width = len(right_header)
    for name, value_lines in rows:
        left_width  = max(left_width, len(name))
        for line in value_lines:
            right_width = max(right_width, len(line))

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

def format_report(report: PipelineComponentExecutionReport, previous_rows: list[tuple[str, list[str]]] | None = None) -> tuple[str, list[tuple[str, list[str]]]]:
    """
    Formats the report of one component as a string
    :param report: The PipelineComponentExecutionReport of a single component
    :param previous_rows: The rows containing the output of the previous report
    :returns tuple[str, list[tuple[str, list[str]]]]: the formatted report as a string and the rows that the report (and its predecessors) already covered
    """
    title = _title(report)
    times = _times(report)

    # Table
    left_header, right_header = "Field", "Value"

    rows = _rows(report.component_output, previous_rows) if previous_rows else _rows(report.component_output)
    table, width = _table(rows, left_header, right_header)

    title_underline_width = max(len(title) + 4, width)
    title_underline = '=' * title_underline_width
    centered_title = title.center(title_underline_width)

    output: list[str] = [title_underline, centered_title, title_underline, '']
    if times:
        output.extend(times + [''])
    output.append("Output (PipelineData):")
    output.extend(table)

    if previous_rows is None:
        output_rows = rows[:]
    else:
        previous_rows.extend(rows)
        output_rows = previous_rows
    return ('\n'.join(output), output_rows) # list[str] => str (one line per string in the list)

def format_all_reports(reports: list[PipelineComponentExecutionReport]) -> str:
    info = _info()
    previous_rows = None
    formatted_reports = [info]
    for report in reports:
        formatted_report, rows = format_report(report, previous_rows)
        formatted_reports.append(formatted_report)
        previous_rows = rows
    return "\n\n\n".join(formatted_reports)