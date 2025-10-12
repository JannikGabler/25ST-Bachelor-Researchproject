import math

import jax
import jax.numpy as jnp
import numpy
from packaging.version import Version

from general_data_structures.directional_acyclic_graph.directional_acyclic_graph import (
    DirectionalAcyclicGraph,
)
from general_data_structures.directional_acyclic_graph.directional_acyclic_graph_node import (
    DirectionalAcyclicGraphNode,
)
from general_data_structures.tree.tree import Tree
from general_data_structures.tree.tree_node import TreeNode

from file_handling.result_persistence.save_policy import SavePolicy


class PipelineConfigurationConstants:
    # Namespace for dynamically loaded modules is getting added on demand
    PARSING_EVAL_NAMESPACE: dict[str, object] = {
        "jax": jax,
        "jax.numpy": jnp,
        "math": math,
        "numpy": numpy,
        "Version": Version,
        "Tree": Tree,
        "TreeNode": TreeNode,
        "DirectionalAcyclicGraph": DirectionalAcyclicGraph,
        "DirectionalAcyclicGraphNode": DirectionalAcyclicGraphNode,
    }


class OldInterpolantsPlotComponentConstants:
    AMOUNT_OF_EVALUATION_POINTS: int = 250

    Y_LIMIT_FACTOR: float = 1.0

    COLORS = [
        "black",
        "#66c2a5",  # grünlich
        "#fc8d62",  # orange
        "#8da0cb",  # bläulich
        "#e78ac3",  # pink
        "#a6d854",  # hellgrün
        "#ffd92f",  # gelb
        "#e5c494",  # beige
        "#b3b3b3",  # grau
    ]

    LINE_STYLES = [
        # '-',  # durchgezogen (solid)
        "--",  # gestrichelt (dashed)
        "-.",  # strich-punkt (dashdot)
        ":",  # gepunktet (dotted)
        (0, (1, 1)),  # sehr feine Punkte
        (0, (5, 5)),  # lange Striche mit Lücken
        (0, (3, 5, 1, 5)),  # Striche mit feinen Punkten
    ]

    SHOW_PLOT_IN_SEPARATE_PROCESS: bool = True

    FIGURE_SIZE = (10, 6)


class InterpolantsPlotComponentConstants:
    AMOUNT_OF_EVALUATION_POINTS: int = 800
    AMOUNT_OF_INF_SCATTER_POINTS: int = 150

    Y_LIMIT_FACTOR: float = 1.0

    COLORS = [
        "black",
        # "#FF0000",  # Rot
        "#00FF00",  # Grün
        "#00BFFF",  # DeepSkyBlue
        # "#FFFF00",  # Gelb
        # "#e5c494",  # beige
        # "brown",
        "#00FFFF",  # Cyan
        "#FF00FF",  # Magenta
        "#FFA500",  # Orange
        "#66c2a5",  # grünlich
        "#8da0cb",  # bläulich
        "#e78ac3",  # pink
        "#a6d854",  # hellgrün
        "#b3b3b3",  # grau
    ]

    LINE_STYLE_DASH_DISTANCE: float = 2.5

    SHOW_PLOT_IN_SEPARATE_PROCESS: bool = True

    FIGURE_SIZE = (10, 6)


class AbsoluteErrorPlotComponentConstants:
    AMOUNT_OF_EVALUATION_POINTS: int = 800

    LINE_WIDTH: int = 2

    COLORS = [
        # "#FF0000",  # Rot
        "#00FF00",  # Grün
        "#00BFFF",  # DeepSkyBlue
        # "#FFFF00",  # Gelb
        # "#e5c494",  # beige
        # "brown",
        "#00FFFF",  # Cyan
        "#FF00FF",  # Magenta
        "#FFA500",  # Orange
        "#66c2a5",  # grünlich
        "#8da0cb",  # bläulich
        "#e78ac3",  # pink
        "#a6d854",  # hellgrün
        "#b3b3b3",  # grau
    ]

    LINE_STYLES = [
        # '-',  # durchgezogen (solid)
        "--",  # gestrichelt (dashed)
        "-.",  # strich-punkt (dashdot)
        ":",  # gepunktet (dotted)
        (0, (1, 1)),  # sehr feine Punkte
        (0, (5, 5)),  # lange Striche mit Lücken
        (0, (3, 5, 1, 5)),  # Striche mit feinen Punkten
    ]

    SHOW_PLOT_IN_SEPARATE_PROCESS: bool = True


class RelativeErrorPlotComponentConstants:
    AMOUNT_OF_EVALUATION_POINTS: int = 800

    LINE_WIDTH: int = 2

    COLORS = [
        # "#FF0000",  # Rot
        "#00FF00",  # Grün
        "#00BFFF",  # DeepSkyBlue
        # "#FFFF00",  # Gelb
        # "#e5c494",  # beige
        # "brown",
        "#00FFFF",  # Cyan
        "#FF00FF",  # Magenta
        "#FFA500",  # Orange
        "#66c2a5",  # grünlich
        "#8da0cb",  # bläulich
        "#e78ac3",  # pink
        "#a6d854",  # hellgrün
        "#b3b3b3",  # grau
    ]

    LINE_STYLES = [
        # '-',  # durchgezogen (solid)
        "--",  # gestrichelt (dashed)
        "-.",  # strich-punkt (dashdot)
        ":",  # gepunktet (dotted)
        (0, (1, 1)),  # sehr feine Punkte
        (0, (5, 5)),  # lange Striche mit Lücken
        (0, (3, 5, 1, 5)),  # Striche mit feinen Punkten
    ]

    SHOW_PLOT_IN_SEPARATE_PROCESS: bool = True


class BaseRoundOffErrorPlotComponentConstants:
    AMOUNT_OF_EVALUATION_POINTS: int = 800

    DEFAULT_Y_LIMIT: float = 10.0
    Y_LIMIT_ATTRIBUTE_NAME: str = "y_limit"
    DEFAULT_Y_THRESHOLD: float = 1.0
    Y_THRESHOLD_ATTRIBUTE_NAME: str = "y_threshold"

    COLORS = [
        # "#FF0000",  # Rot
        "#00FF00",  # Grün
        "#00BFFF",  # DeepSkyBlue
        # "#FFFF00",  # Gelb
        # "#e5c494",  # beige
        # "brown",
        "#00FFFF",  # Cyan
        "#FF00FF",  # Magenta
        "#FFA500",  # Orange
        "#66c2a5",  # grünlich
        "#8da0cb",  # bläulich
        "#e78ac3",  # pink
        "#a6d854",  # hellgrün
        "#b3b3b3",  # grau
    ]

    LINE_WIDTH: int = 2
    LINE_STYLE_DASH_DISTANCE: float = 1

    SHOW_PLOT_IN_SEPARATE_PROCESS: bool = True

    FIGURE_SIZE = (10, 6)


class AbsoluteRoundOffErrorPlotComponentConstants:
    SHOW_PLOT_IN_SEPARATE_PROCESS: bool = True


class RelativeRoundOffErrorPlotComponentConstants:
    SHOW_PLOT_IN_SEPARATE_PROCESS: bool = True


class FilesystemResultStoreConstants:
    POLICY = SavePolicy(
        mode="soft-state",  # or "snapshot"
        keep_soft_state_n=3,
        json_indent=2,
        plot_formats=("svg", "png", "pdf"),
    )
