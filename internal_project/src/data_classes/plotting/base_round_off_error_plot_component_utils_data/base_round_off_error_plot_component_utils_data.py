import jax.numpy as jnp

from dataclasses import dataclass
from fractions import Fraction

from functions.abstracts.compilable_function import CompilableFunction


@dataclass
class BaseRoundOffErrorPlotComponentUtilsData:
    """
    Container for all data related to round-off error visualization within the pipeline.
    This dataclass provides exact and approximate evaluation data, interpolation nodes, barycentric weights, and
    computed round-off errors, as well as metadata required for plotting and analysis of numerical stability.

    Attributes:
        amount_of_functions_to_plot (int | None): Number of functions to include in the round-off error plots.
        evaluation_points (jnp.ndarray | None): Evaluation points used for numerical computations.
        evaluation_points_exact (list[Fraction] | None): Exact representations of the evaluation points as fractions.
        interpolation_nodes_exact (list[Fraction] | None): Exact x-coordinates of interpolation nodes.
        interpolation_values_exact (list[Fraction] | None): Exact function values at the interpolation nodes.
        barycentric_weights_exact (list[Fraction] | None): Exact barycentric weights used for interpolation.
        interpolant_values_exact (list[Fraction] | None): Exact interpolant values at the evaluation points.
        functions (list[CompilableFunction] | None): List of functions evaluated for round-off error analysis.
        function_names (list[str] | None): Human-readable names of the functions corresponding to `functions`.
        round_off_errors (list[jnp.ndarray] | None): Computed round-off errors for each function.
        y_threshold (float | None): Threshold used to highlight or clip error values in plots.
        y_limit (float | None): Maximum y-axis limit applied to round-off error plots.
        scatter_size (float | None): Size of scatter points when visualizing discrete error samples.
        scatter_x_distance (jnp.ndarray | None): Horizontal distance between neighboring scatter points.
        scatter_y_distance (jnp.ndarray | None): Vertical distance between neighboring scatter points.
        border (tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]] | None):
            Coordinates defining upper and lower border lines in the plot area.
        connectable_segments (list[tuple[list[list[int]], list[list[int]]]] | None):
            Segments describing which plot elements or regions are connectable in visualization.
    """

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
    border: (tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]] | None) = None
    connectable_segments: list[tuple[list[list[int]], list[list[int]]]] | None = None
