import jax.numpy as jnp

from dataclasses import dataclass
from fractions import Fraction

from functions.abstracts.compilable_function import CompilableFunction


@dataclass
class BaseRoundOffErrorPlotComponentUtilsData:
    amount_of_functions_to_plot: int | None = None

    evaluation_points: jnp.ndarray | None = None
    evaluation_points_exact: list[Fraction] | None = None

    interpolation_nodes_exact: list[Fraction] | None = None
    interpolation_values_exact: list[Fraction] | None = None

    barycentric_weights_exact: list[Fraction] | None = None
    interpolant_values_exact: list[Fraction] | None = None

    functions: list[CompilableFunction] | None = None
    function_names: list[str] | None = None

    round_off_errors: list[jnp.ndarray] | None = None

    y_threshold: float | None = None
    y_limit: float | None = None
    scatter_size: float | None = None
    scatter_x_distance: jnp.ndarray | None = None
    scatter_y_distance: jnp.ndarray | None = None
    border: (
        tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]] | None
    ) = None
    connectable_segments: list[tuple[list[list[int]], list[list[int]]]] | None = None
