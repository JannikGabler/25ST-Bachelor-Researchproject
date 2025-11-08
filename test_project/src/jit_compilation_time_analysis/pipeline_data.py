from dataclasses import dataclass
from typing import Optional, Tuple
import jax.numpy as jnp


@dataclass
class PipelineData:
    node_count: Optional[int]
    interpolation_interval: Optional[Tuple[float, float]]
    data_type: Optional[type]
    function_values: Optional[jnp.ndarray] = None

    node_array: Optional[jnp.ndarray] = None
