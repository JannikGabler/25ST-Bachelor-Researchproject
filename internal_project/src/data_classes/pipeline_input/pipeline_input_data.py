from dataclasses import dataclass, field


@dataclass
class PipelineInputData:
    """
    Input container for pipeline execution. This dataclass provides all user-provided inputs that define functions, interpolation parameters,
    evaluation points, and optional additional values.

    Attributes:
        name (str | None): Optional name of the pipeline run.
        data_type (str | None): Data type to be used.
        node_count (str | None): Number of interpolation nodes.
        interpolation_interval (str | None): Interval over which interpolation is performed.
        function_expression (str | None): String representation of a function expression.
        piecewise_function_expression (str | None): String encoding a piecewise-defined function.
        sympy_function_expression_simplification (str | None): Flag to enable or disable Sympy simplification.
        function_callable (str | None): Reference or identifier of a callable function provided by the user.
        interpolation_values (str | None): String-encoded list or array of interpolation values.
        interpolant_evaluation_points (str | None): String-encoded points at which the interpolant should be evaluated.
        additional_directly_injected_values (dict[str, str]): Additional values injected directly into pipeline data.
        additional_values (dict[str, str]): Dictionary for further values.
    """

    name: str | None = None

    data_type: str | None = None
    node_count: str | None = None
    interpolation_interval: str | None = None

    function_expression: str | None = None
    piecewise_function_expression: str | None = None
    sympy_function_expression_simplification: str | None = None
    function_callable: str | None = None
    interpolation_values: str | None = None

    interpolant_evaluation_points: str | None = None

    additional_directly_injected_values: dict[str, str] = field(default_factory=dict)
    additional_values: dict[str, str] = field(default_factory=dict)
