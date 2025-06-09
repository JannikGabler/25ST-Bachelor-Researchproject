from dataclasses import dataclass
from typing import Optional, Tuple
import jax.numpy as jnp

@dataclass
class PipelineData:
    node_count: Optional[int]
    data_type: Optional[type]
    interpolation_interval: Optional[jnp.ndarray] = None # Warning: Changed!
    function_values: Optional[jnp.ndarray] = None

    nodes: Optional[jnp.ndarray] = None





