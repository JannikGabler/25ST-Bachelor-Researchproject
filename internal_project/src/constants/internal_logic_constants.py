import math

import jax
import jax.numpy as jnp
import numpy
from packaging.version import Version

from general_data_structures.directional_acyclic_graph.directional_acyclic_graph import DirectionalAcyclicGraph
from general_data_structures.directional_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode
from general_data_structures.tree.tree import Tree
from general_data_structures.tree.tree_node import TreeNode


class PipelineConfigurationConstants:
    # Namespace for dynamically loaded modules is getting added on demand
    PARSING_EVAL_NAMESPACE: dict[str, object] = {'jax': jax, 'jax.numpy': jnp, 'math': math, 'numpy': numpy,
                                                'Version': Version, 'Tree': Tree, 'TreeNode': TreeNode,
                                                'DirectionalAcyclicGraph': DirectionalAcyclicGraph,
                                                'DirectionalAcyclicGraphNode': DirectionalAcyclicGraphNode}



class InterpolantsPlotComponentConstants:
    AMOUNT_OF_EVALUATION_POINTS: int = 100

    Y_LIMIT_FACTOR: float = 2.0

    COLORS = [
        "black",
        "#66c2a5",  # grünlich
        "#fc8d62",  # orange
        "#8da0cb",  # bläulich
        "#e78ac3",  # pink
        "#a6d854",  # hellgrün
        "#ffd92f",  # gelb
        "#e5c494",  # beige
        "#b3b3b3"  # grau
    ]

    LINE_STYLES = [
        # '-',  # durchgezogen (solid)
        '--',  # gestrichelt (dashed)
        '-.',  # strich-punkt (dashdot)
        ':',  # gepunktet (dotted)
        (0, (1, 1)),  # sehr feine Punkte
        (0, (5, 5)),  # lange Striche mit Lücken
        (0, (3, 5, 1, 5)),  # Striche mit feinen Punkten
    ]

    SHOW_PLOT_IN_SEPARATE_PROCESS: bool = True



class AbsoluteErrorPlotComponentConstants:
    AMOUNT_OF_EVALUATION_POINTS: int = 100

    LINE_WIDTH: int = 2

    COLORS = [
        "#66c2a5",  # grünlich
        "#fc8d62",  # orange
        "#8da0cb",  # bläulich
        "#e78ac3",  # pink
        "#a6d854",  # hellgrün
        "#ffd92f",  # gelb
        "#e5c494",  # beige
        "#b3b3b3"  # grau
        "black",
    ]

    LINE_STYLES = [
        # '-',  # durchgezogen (solid)
        '--',  # gestrichelt (dashed)
        '-.',  # strich-punkt (dashdot)
        ':',  # gepunktet (dotted)
        (0, (1, 1)),  # sehr feine Punkte
        (0, (5, 5)),  # lange Striche mit Lücken
        (0, (3, 5, 1, 5)),  # Striche mit feinen Punkten
    ]

    SHOW_PLOT_IN_SEPARATE_PROCESS: bool = True



class RelativeErrorPlotComponentConstants:
    AMOUNT_OF_EVALUATION_POINTS: int = 100

    LINE_WIDTH: int = 2

    COLORS = [
        "#66c2a5",  # grünlich
        "#fc8d62",  # orange
        "#8da0cb",  # bläulich
        "#e78ac3",  # pink
        "#a6d854",  # hellgrün
        "#ffd92f",  # gelb
        "#e5c494",  # beige
        "#b3b3b3"  # grau
        "black",
    ]

    LINE_STYLES = [
        # '-',  # durchgezogen (solid)
        '--',  # gestrichelt (dashed)
        '-.',  # strich-punkt (dashdot)
        ':',  # gepunktet (dotted)
        (0, (1, 1)),  # sehr feine Punkte
        (0, (5, 5)),  # lange Striche mit Lücken
        (0, (3, 5, 1, 5)),  # Striche mit feinen Punkten
    ]

    SHOW_PLOT_IN_SEPARATE_PROCESS: bool = True