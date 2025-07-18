from dataclasses import dataclass, field
from typing import Optional, Any, Callable
import jax.numpy as jnp

from interpolants.abstracts.compilable_interpolant import CompilableInterpolant


@dataclass
class PipelineData:
    data_type: jnp.dtype | None = None
    node_count: int | None = None
    interpolation_interval: jnp.ndarray | None = None
    interpolant_evaluation_points: jnp.ndarray | None = None

    function_callable: Callable[[jnp.ndarray], jnp.ndarray] | None = None   # Not Jax compiled, must be compiled by using components

    interpolation_nodes: jnp.ndarray | None = None
    interpolation_values: jnp.ndarray | None = None

    interpolant: Optional[CompilableInterpolant] = None
    interpolant_values: jnp.ndarray | None = None

    additional_values: dict[str, Any] = field(default_factory=dict)






