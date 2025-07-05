from dataclasses import dataclass, field


@dataclass
class PipelineInputData:
    name: str | None = None

    data_type: str | None = None
    node_count: str | None = None
    interpolation_interval: str | None = None

    function_expression: str | None = None
    piecewise_function_expression: str | None = None
    sympy_function_expression_simplification: str | None = None
    function_callable: str | None = None
    interpolation_values: str | None = None

    additional_directly_injected_values: dict[str, str] = field(default_factory=dict)
    additional_values: dict[str, str] = field(default_factory=dict)



