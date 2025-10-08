import jax.numpy as jnp
from jax.typing import DTypeLike

from dataclasses import dataclass, field
from typing import Optional, Any, Callable

from data_classes.plot_template.plot_template import PlotTemplate
from functions.abstracts.compilable_function import CompilableFunction


@dataclass
class PipelineData:
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
