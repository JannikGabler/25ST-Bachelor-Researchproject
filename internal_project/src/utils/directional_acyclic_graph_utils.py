from __future__ import annotations
from typing import TYPE_CHECKING

from copy import copy
from typing import TypeVar, Generator, Any

if TYPE_CHECKING:
    from general_data_structures.directional_acyclic_graph.directional_acyclic_graph import DirectionalAcyclicGraph
    from general_data_structures.directional_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode


from exceptions.not_instantiable_error import NotInstantiableError
from utils.collections_utils import CollectionsUtils

T = TypeVar("T")


class DirectionalAcyclicGraphUtils:
    """
    Utility helpers for DAG operations. This class is not meant to be instantiated.
    """

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        """
        Raises:
            NotInstantiableError: Always raised when initialized to indicate that this class is not meant to be instantiated.
        """

        raise NotInstantiableError("The class 'JaxUtils' can not be instantiated.")


    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def get_entry_nodes_to_node_graph(nodes: list[DirectionalAcyclicGraphNode[T]]) -> list[DirectionalAcyclicGraphNode[T]]:
        """
        Collect all entry nodes (nodes without predecessors) reachable from the given nodes.

        Args:
            nodes (list[DirectionalAcyclicGraphNode[T]]): Starting nodes.

        Returns:
            list[DirectionalAcyclicGraphNode[T]]: Entry nodes.
        """

        stack: list[DirectionalAcyclicGraphNode[T]] = copy(nodes)
        entry_nodes: list[DirectionalAcyclicGraphNode[T]] = []

        while stack:
            current_node = stack.pop()
            predecessors = current_node.predecessors

            if predecessors:
                stack.extend(predecessors)
            elif not CollectionsUtils.is_exact_element_in_collection(current_node, entry_nodes):
                entry_nodes.append(current_node)

        return entry_nodes


    @staticmethod
    def topological_traversal(argument: DirectionalAcyclicGraph[T] | DirectionalAcyclicGraphNode[T]) -> Generator[DirectionalAcyclicGraphNode[T], Any, None]:
        """
        Yield nodes in topological order (Kahn’s algorithm).

        Args:
            argument (DirectionalAcyclicGraph[T] | DirectionalAcyclicGraphNode[T]): Graph or starting node.

        Returns:
            Generator[DirectionalAcyclicGraphNode[T], Any, None]: Nodes in topological order.
        """

        in_degree: dict[int, int] = {}
        node_lookup: dict[int, DirectionalAcyclicGraphNode[T]] = {}

        DirectionalAcyclicGraphUtils._populate_in_degree_and_id_lookup_dicts_(argument, in_degree, node_lookup)
        zero_in_degree_nodes = [node_lookup[node_id] for node_id, deg in in_degree.items() if deg == 0]

        yield from DirectionalAcyclicGraphUtils._kahn_algorithm_main_(in_degree, zero_in_degree_nodes)


    @staticmethod
    def ascii_dag(adj: dict, labels: dict) -> str:
        """
        Draw an ASCII representation of a DAG rotated 90° clockwise (vertical growth).

        Args:
            adj (dict): Adjacency dictionary, e.g. {node: [successor1, successor2, ...]}.
            labels (dict): Labels dictionary, e.g. {node: "label"}.

        Returns:
            str: ASCII diagram.
        """

        rev = {u: [] for u in adj}
        for u, vs in adj.items():
            for v in vs:
                rev.setdefault(v, []).append(u)

        levels = {}

        def compute_level(u):
            if u in levels:
                return levels[u]
            if not rev.get(u):
                levels[u] = 0
            else:
                levels[u] = max(compute_level(p) for p in rev[u]) + 1
            return levels[u]

        for u in adj:
            compute_level(u)

        layers = {}
        for u, lvl in levels.items():
            layers.setdefault(lvl, []).append(u)
        for lvl in layers:
            layers[lvl].sort(key=lambda n: str(n))

        max_label_len = max(len(labels[u]) for u in adj)
        box_w = max_label_len + 4  # padding
        box_h = 3
        h_spacing = 4
        v_spacing = 1

        coords = {}
        for lvl, nodes in layers.items():
            for idx, u in enumerate(nodes):
                x = idx * (box_w + h_spacing)
                y = lvl * (box_h + v_spacing)
                coords[u] = (x, y)

        width = max(x + box_w for x, y in coords.values()) + 1
        height = max(y + box_h for x, y in coords.values()) + 1

        canvas = [[" " for _ in range(width)] for _ in range(height)]

        for u, (x, y) in coords.items():
            label = labels[u]
            top = y
            mid = y + 1
            bot = y + 2

            for i in range(box_w):
                canvas[top][x + i] = "="
                canvas[bot][x + i] = "="

            text = f"  {label}  ".center(box_w)
            canvas[mid][x] = canvas[mid][x + box_w - 1] = "="
            for i, ch in enumerate(text):
                canvas[mid][x + 1 + i] = ch

        arrow_margin = 1
        for u, vs in adj.items():
            x0, y0 = coords[u]
            y0r = y0 + box_h
            x0m = x0 + box_w // 2
            for v in vs:
                x1, y1 = coords[v]
                y1t = y1
                x1m = x1 + box_w // 2

                for y in range(y0r + arrow_margin, y1t - arrow_margin):
                    if canvas[y][x0m] == " ":
                        canvas[y][x0m] = "|"

                arrow_y = y1t - arrow_margin
                canvas[arrow_y][x0m] = "v"

                if x0m < x1m:
                    for xx in range(x0m + 1, x1m):
                        if canvas[arrow_y][xx] == " ":
                            canvas[arrow_y][xx] = "-"
                    canvas[arrow_y][x1m] = ">"
                elif x0m > x1m:
                    for xx in range(x1m + 1, x0m):
                        if canvas[arrow_y][xx] == " ":
                            canvas[arrow_y][xx] = "-"
                    canvas[arrow_y][x1m] = "<"

        for u, (x, y) in coords.items():
            mid = y + 1
            canvas[mid][x] = "="
            canvas[mid][x + box_w - 1] = "="

        return "\n".join("".join(row) for row in canvas)


    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _populate_in_degree_and_id_lookup_dicts_(argument: DirectionalAcyclicGraph[T] | DirectionalAcyclicGraphNode[T], in_degree: dict[int, int], node_lookup: dict[int, DirectionalAcyclicGraphNode[T]]) -> None:
        for node in argument.depth_first_traversal():
            node_id = id(node)
            in_degree[node_id] = len(node.predecessors)
            node_lookup[node_id] = node


    @staticmethod
    def _kahn_algorithm_main_(in_degree: dict[int, int], zero_in_degree_nodes: list[DirectionalAcyclicGraphNode[T]]) -> Generator[DirectionalAcyclicGraphNode[T], Any, None]:
        while zero_in_degree_nodes:
            current_node = zero_in_degree_nodes.pop()
            yield current_node

            for successor in current_node.successors:
                succ_id = id(successor)
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    zero_in_degree_nodes.append(successor)
