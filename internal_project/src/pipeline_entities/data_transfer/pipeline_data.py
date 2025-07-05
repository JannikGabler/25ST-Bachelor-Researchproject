from dataclasses import dataclass, field
from typing import Optional, Any, Callable
import jax.numpy as jnp


@dataclass
class PipelineData:
    data_type: type | None = None
    node_count: int | None = None
    interpolation_interval: jnp.ndarray | None = None

    function_callable: Callable[[jnp.ndarray], jnp.ndarray] | None = None   # Not Jax compiled, must be compiled by using components

    interpolation_nodes: jnp.ndarray | None = None
    interpolation_values: jnp.ndarray | None = None

    interpolant: jnp.ndarray | None = None

    additional_values: dict[str, Any] = field(default_factory=dict)




