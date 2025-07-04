from dataclasses import dataclass, field
from typing import Optional, Any, Callable
import jax.numpy as jnp

from data_structures.interpolants.abstracts.interpolant import Interpolant


@dataclass
class PipelineData:
    data_type: Optional[type] = None

    node_count: Optional[int] = None
    interpolation_interval: Optional[jnp.ndarray] = None
    function_callable: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    function_values: Optional[jnp.ndarray] = None


    nodes: Optional[jnp.ndarray] = None

    interpolant: Optional[Interpolant] = None

    additional_values: dict[str, Any] = field(default_factory=dict)






