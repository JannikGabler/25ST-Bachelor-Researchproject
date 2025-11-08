import jax.numpy as jnp
from jax.typing import DTypeLike

from dataclasses import dataclass, field
from typing import Optional, Any, Callable

from data_classes.plot_template.plot_template import PlotTemplate
from functions.abstracts.compilable_function import CompilableFunction


@dataclass
class PipelineData:
    """
    Container for all data related to interpolation tasks within the pipeline. This dataclass provides input functions, interpolation nodes, values,
    and results of computed interpolants. It also supports additional metadata required for evaluation, plotting, or further processing.

    Attributes:
        data_type (DTypeLike | None): The numerical data type used for all arrays in the pipeline.
        node_count (int | None): Number of interpolation nodes.
        interpolation_interval (jnp.ndarray | None): Interval over which interpolation is performed.
        interpolant_evaluation_points (jnp.ndarray | None): Points at which the interpolant should be evaluated.
        original_function (CompilableFunction | None): The original function to be interpolated.
        interpolation_nodes (jnp.ndarray | None): The x-coordinates of the interpolation nodes.
        interpolation_values (jnp.ndarray | None): The y-values of the original function at the interpolation nodes.
        interpolant (CompilableFunction | None): The constructed interpolant function.
        interpolant_values (jnp.ndarray | None): Values of the interpolant evaluated at the specified evaluation points.
        additional_values (dict[str, Any]): A flexible dictionary for storing additional values.
    """

    data_type: DTypeLike | None = None
    node_count: int | None = None
    interpolation_interval: jnp.ndarray | None = None
    interpolant_evaluation_points: jnp.ndarray | None = None

    original_function: CompilableFunction | None = None

    interpolation_nodes: jnp.ndarray | None = None
    interpolation_values: jnp.ndarray | None = None

    interpolant: CompilableFunction | None = None
    interpolant_values: jnp.ndarray | None = None

    plots: list[PlotTemplate] | None = None

    additional_values: dict[str, Any] = field(default_factory=dict)
