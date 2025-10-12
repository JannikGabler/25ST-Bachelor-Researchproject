import jax.numpy as jnp

from dataclasses import dataclass


@dataclass
class FunctionPlotData:
    function_name: str
    function_index: int

    connectable_segments: list[list[tuple[jnp.ndarray, jnp.ndarray]]]

    single_points: list[tuple[jnp.ndarray, jnp.ndarray]]
