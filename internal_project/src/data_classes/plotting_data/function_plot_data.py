import jax.numpy as jnp

from dataclasses import dataclass


@dataclass
class FunctionPlotData:
    """
    Container for plot-ready data of a function. This dataclass provides all information needed to render a function in a plot,
    including its name, index, connectable line segments, and isolated points.

    Attributes:
        function_name (str): The display name of the function.
        function_index (int): The function index.
        connectable_segments (list[list[tuple[jnp.ndarray, jnp.ndarray]]]):
            A list of line segments where each segment is defined by pairs of x and y arrays that can be connected.
        single_points (list[tuple[jnp.ndarray, jnp.ndarray]]):
            A list of individual points defined by x and y arrays that are not part of a segment.
    """

    function_name: str
    function_index: int

    connectable_segments: list[list[tuple[jnp.ndarray, jnp.ndarray]]]

    single_points: list[tuple[jnp.ndarray, jnp.ndarray]]
