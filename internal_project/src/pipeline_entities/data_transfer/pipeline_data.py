from dataclasses import dataclass, field
from typing import Optional, Tuple, Any
import jax.numpy as jnp

@dataclass
class PipelineData:
    data_type: Optional[type] = None

    node_count: Optional[int] = None
    interpolation_interval: Optional[jnp.ndarray] = None
    function_values: Optional[jnp.ndarray] = None

    nodes: Optional[jnp.ndarray] = None

    interpolant: Optional[jnp.ndarray] = None

    additional_values: dict[str, Any] = field(default_factory=dict)






