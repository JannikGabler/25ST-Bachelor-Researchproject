from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass
class PlotOperationData:
    x_points: object
    y_points: object

    alpha: float | None = None
    color: str | None = None
    label: str | None = None
    zorder: float | None = None
