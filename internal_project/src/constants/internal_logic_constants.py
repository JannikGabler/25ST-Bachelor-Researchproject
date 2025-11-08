import math

import jax
import jax.numpy as jnp
import numpy
from packaging.version import Version

from general_data_structures.directional_acyclic_graph.directional_acyclic_graph import DirectionalAcyclicGraph
from general_data_structures.directional_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode
from general_data_structures.tree.tree import Tree
from general_data_structures.tree.tree_node import TreeNode

from file_handling.result_persistence.save_policy import SavePolicy


class PipelineConfigurationConstants:
    """
    Constants used for pipeline configuration, including a namespace for dynamically loaded modules.
    """


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
    """
    Constants defining default parameters for the old interpolants plot component, including evaluation points, colors, line styles, and plotting options.
    """


    AMOUNT_OF_EVALUATION_POINTS: int = 250

    Y_LIMIT_FACTOR: float = 1.0

    COLORS = [
        "black",
        "#66c2a5",  # greenish
        "#fc8d62",  # orange
        "#8da0cb",  # blueish
        "#e78ac3",  # pink
        "#a6d854",  # light green
        "#ffd92f",  # yellow
        "#e5c494",  # beige
        "#b3b3b3",  # grey
    ]

    LINE_STYLES = [
        # '-',  # solid
        "--",  # dashed
        "-.",  # dashdot
        ":",  # dotted
        (0, (1, 1)),  # very light dots
        (0, (5, 5)),  # long lines with gaps
        (0, (3, 5, 1, 5)),  # lines with light dots
    ]

    SHOW_PLOT_IN_SEPARATE_PROCESS: bool = True

    FIGURE_SIZE = (10, 6)


class InterpolantsPlotComponentConstants:
    """
    Constants defining default parameters for interpolants plot component, including evaluation points, axis limits, colors, line styles, and plotting options.
    """


    AMOUNT_OF_EVALUATION_POINTS: int = 800
    AMOUNT_OF_INF_SCATTER_POINTS: int = 150

    Y_LIMIT_FACTOR: float = 1.0

    COLORS = [
        "black",
        # "#FF0000",  # red
        "#00FF00",  # green
        "#00BFFF",  # DeepSkyBlue
        # "#FFFF00",  # yellow
        # "#e5c494",  # beige
        # "brown",
        "#00FFFF",  # Cyan
        "#FF00FF",  # Magenta
        "#FFA500",  # Orange
        "#66c2a5",  # greenish
        "#8da0cb",  # blueish
        "#e78ac3",  # pink
        "#a6d854",  # light green
        "#b3b3b3",  # grey
    ]

    LINE_STYLE_DASH_DISTANCE: float = 2.5

    SHOW_PLOT_IN_SEPARATE_PROCESS: bool = True

    FIGURE_SIZE = (10, 6)


class AbsoluteErrorPlotComponentConstants:
    """
    Constants defining default parameters for absolute error plot component, including evaluation points, line width, colors, line styles, and plotting options.
    """


    AMOUNT_OF_EVALUATION_POINTS: int = 800

    LINE_WIDTH: int = 2

    COLORS = [
        # "#FF0000",  # red
        "#00FF00",  # green
        "#00BFFF",  # DeepSkyBlue
        # "#FFFF00",  # yellow
        # "#e5c494",  # beige
        # "brown",
        "#00FFFF",  # Cyan
        "#FF00FF",  # Magenta
        "#FFA500",  # Orange
        "#66c2a5",  # greenish
        "#8da0cb",  # blueish
        "#e78ac3",  # pink
        "#a6d854",  # light green
        "#b3b3b3",  # grey
    ]

    LINE_STYLES = [
        # '-',  # solid
        "--",  # dashed
        "-.",  # dashdot
        ":",  # dotted
        (0, (1, 1)),  # very light dots
        (0, (5, 5)),  # long lines with gaps
        (0, (3, 5, 1, 5)),  # lines with light dots
    ]

    SHOW_PLOT_IN_SEPARATE_PROCESS: bool = True


class RelativeErrorPlotComponentConstants:
    """
    Constants defining default parameters for relative error plot component, including evaluation points, line width, colors, line styles, and plotting options.
    """


    AMOUNT_OF_EVALUATION_POINTS: int = 800

    LINE_WIDTH: int = 2

    COLORS = [
        # "#FF0000",  # red
        "#00FF00",  # green
        "#00BFFF",  # DeepSkyBlue
        # "#FFFF00",  # yellow
        # "#e5c494",  # beige
        # "brown",
        "#00FFFF",  # Cyan
        "#FF00FF",  # Magenta
        "#FFA500",  # Orange
        "#66c2a5",  # greenish
        "#8da0cb",  # blueish
        "#e78ac3",  # pink
        "#a6d854",  # light green
        "#b3b3b3",  # grey
    ]

    LINE_STYLES = [
        # '-',  # solid
        "--",  # dashed
        "-.",  # dashdot
        ":",  # dotted
        (0, (1, 1)),
        (0, (5, 5)),
        (0, (3, 5, 1, 5)),
    ]

    SHOW_PLOT_IN_SEPARATE_PROCESS: bool = True


class BaseRoundOffErrorPlotComponentConstants:
    """
    Constants defining default parameters for base round-off error plot components, including evaluation points, axis limits, colors, and line styles.
    """


    AMOUNT_OF_EVALUATION_POINTS: int = 800

    DEFAULT_Y_LIMIT: float = 10.0
    Y_LIMIT_ATTRIBUTE_NAME: str = "y_limit"
    DEFAULT_Y_THRESHOLD: float = 1.0
    Y_THRESHOLD_ATTRIBUTE_NAME: str = "y_threshold"

    COLORS = [
        # "#FF0000",
        "#00FF00",
        "#00BFFF",
        # "#FFFF00",
        # "#e5c494",
        # "brown",
        "#00FFFF",
        "#FF00FF",
        "#FFA500",
        "#66c2a5",
        "#8da0cb",
        "#e78ac3",
        "#a6d854",
        "#b3b3b3",
    ]

    LINE_WIDTH: int = 2
    LINE_STYLE_DASH_DISTANCE: float = 1

    SHOW_PLOT_IN_SEPARATE_PROCESS: bool = True

    FIGURE_SIZE = (10, 6)


class AbsoluteRoundOffErrorPlotComponentConstants:
    """
    Constants defining default parameters for absolute round-off error plot components, including plotting options and process configuration.
    """


    SHOW_PLOT_IN_SEPARATE_PROCESS: bool = True


class RelativeRoundOffErrorPlotComponentConstants:
    """
    Constants defining default parameters for relative round-off error plot components, including plotting options and process configuration.
    """


    SHOW_PLOT_IN_SEPARATE_PROCESS: bool = True


class FilesystemResultStoreConstants:
    """
    Constants defining default save policy and file formats for filesystem-based result storage.
    """


    POLICY = SavePolicy(
        mode="soft-state",  # or "snapshot"
        keep_soft_state_n=3,
        json_indent=2,
        plot_formats=("svg", "png", "pdf"),
    )
