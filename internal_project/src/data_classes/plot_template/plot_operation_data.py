from dataclasses import dataclass


@dataclass
class PlotOperationData:
    """
    Container for data used in a single plot operation.
    This dataclass defines the x- and y-coordinates of plotted points along with optional styling and rendering attributes.

    Attributes:
        x_points (object): X-coordinates of the data points to be plotted.
        y_points (object): Y-coordinates of the data points to be plotted.
        alpha (float | None): Opacity level of the plot elements (0.0–1.0).
        color (str | None): Color used for the plot elements.
        label (str | None): Label for legend or identification in the plot.
        zorder (float | None): Drawing order of the plot elements relative to others.
    """

    x_points: object
    y_points: object

    alpha: float | None = None
    color: str | None = None
    label: str | None = None
    zorder: float | None = None
