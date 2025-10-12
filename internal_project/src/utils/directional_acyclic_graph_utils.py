from __future__ import annotations
from typing import TYPE_CHECKING

from copy import copy
from typing import TypeVar, Generator, Any

if TYPE_CHECKING:
    from general_data_structures.directional_acyclic_graph.directional_acyclic_graph import (
        DirectionalAcyclicGraph,
    )
    from general_data_structures.directional_acyclic_graph.directional_acyclic_graph_node import (
        DirectionalAcyclicGraphNode,
    )


from exceptions.not_instantiable_error import NotInstantiableError
from utils.collections_utils import CollectionsUtils

T = TypeVar("T")


class DirectionalAcyclicGraphUtils:
    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        raise NotInstantiableError("The class 'JaxUtils' can not be instantiated.")

    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def get_entry_nodes_to_node_graph(
        nodes: list[DirectionalAcyclicGraphNode[T]],
    ) -> list[DirectionalAcyclicGraphNode[T]]:
        stack: list[DirectionalAcyclicGraphNode[T]] = copy(nodes)
        entry_nodes: list[DirectionalAcyclicGraphNode[T]] = []

        while stack:
            current_node = stack.pop()
            predecessors = current_node.predecessors

            if predecessors:
                stack.extend(predecessors)
            elif not CollectionsUtils.is_exact_element_in_collection(
                current_node, entry_nodes
            ):
                entry_nodes.append(current_node)

        return entry_nodes

    @staticmethod
    def topological_traversal(
        argument: DirectionalAcyclicGraph[T] | DirectionalAcyclicGraphNode[T],
    ) -> Generator[DirectionalAcyclicGraphNode[T], Any, None]:
        in_degree: dict[int, int] = {}
        node_lookup: dict[int, DirectionalAcyclicGraphNode[T]] = {}

        DirectionalAcyclicGraphUtils._populate_in_degree_and_id_lookup_dicts_(
            argument, in_degree, node_lookup
        )
        zero_in_degree_nodes = [
            node_lookup[node_id] for node_id, deg in in_degree.items() if deg == 0
        ]

        yield from DirectionalAcyclicGraphUtils._kahn_algorithm_main_(
            in_degree, zero_in_degree_nodes
        )

    @staticmethod
    def ascii_dag(adj: dict, labels: dict) -> str:
        """
        Draw an ASCII representation of a DAG rotated 90° clockwise (vertical growth).
        - adj: dict[node, list of successor nodes]
        - labels: dict[node, str label]
        """
        # Build reverse adjacency for level assignment
        rev = {u: [] for u in adj}
        for u, vs in adj.items():
            for v in vs:
                rev.setdefault(v, []).append(u)

        # Compute levels (distance from sources)
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

        # Group nodes by level
        layers = {}
        for u, lvl in levels.items():
            layers.setdefault(lvl, []).append(u)
        for lvl in layers:
            layers[lvl].sort(key=lambda n: str(n))

        # Calculate box and spacing sizes
        max_label_len = max(len(labels[u]) for u in adj)
        box_w = max_label_len + 4  # padding
        box_h = 3
        h_spacing = 4
        v_spacing = 1

        # Assign coordinates - ROTATION:
        # vorher: x = lvl * (box_w + h_spacing)
        #          y = idx * (box_h + v_spacing)
        # jetzt:  x = idx * (box_w + h_spacing)
        #          y = lvl * (box_h + v_spacing)
        coords = {}
        for lvl, nodes in layers.items():
            for idx, u in enumerate(nodes):
                x = idx * (box_w + h_spacing)
                y = lvl * (box_h + v_spacing)
                coords[u] = (x, y)

        # Canvas size
        width = max(x + box_w for x, y in coords.values()) + 1
        height = max(y + box_h for x, y in coords.values()) + 1

        # Initialize canvas
        canvas = [[" " for _ in range(width)] for _ in range(height)]

        # Draw boxes (unverändert)
        for u, (x, y) in coords.items():
            label = labels[u]
            top = y
            mid = y + 1
            bot = y + 2
            # Top and bottom border
            for i in range(box_w):
                canvas[top][x + i] = "="
                canvas[bot][x + i] = "="
            # Middle content
            text = f"  {label}  ".center(box_w)
            canvas[mid][x] = canvas[mid][x + box_w - 1] = "="
            for i, ch in enumerate(text):
                canvas[mid][x + 1 + i] = ch

        # Draw edges - ROTATION:
        # vorher: Pfeile horizontal von rechts zu links
        # jetzt:  Pfeile vertikal von oben nach unten
        arrow_margin = 1
        for u, vs in adj.items():
            x0, y0 = coords[u]
            y0r = y0 + box_h
            x0m = x0 + box_w // 2  # Mitte horizontal der Box
            for v in vs:
                x1, y1 = coords[v]
                y1t = y1
                x1m = x1 + box_w // 2

                # Vertikale Linie vom unteren Rand von u bis oberer Rand von v
                for y in range(y0r + arrow_margin, y1t - arrow_margin):
                    if canvas[y][x0m] == " ":
                        canvas[y][x0m] = "|"

                # Pfeilspitze (nach unten)
                arrow_y = y1t - arrow_margin
                canvas[arrow_y][x0m] = "v"

                # Horizontale Verbindung wenn x verschoben (zwischen x0m und x1m)
                if x0m < x1m:
                    for xx in range(x0m + 1, x1m):
                        if canvas[arrow_y][xx] == " ":
                            canvas[arrow_y][xx] = "-"
                    # Pfeilspitze nach rechts, falls horizontal weiter
                    canvas[arrow_y][x1m] = ">"
                elif x0m > x1m:
                    for xx in range(x1m + 1, x0m):
                        if canvas[arrow_y][xx] == " ":
                            canvas[arrow_y][xx] = "-"
                    # Pfeilspitze nach links
                    canvas[arrow_y][x1m] = "<"

        # Restore mid-line borders for closed boxes
        for u, (x, y) in coords.items():
            mid = y + 1
            canvas[mid][x] = "="
            canvas[mid][x + box_w - 1] = "="

        return "\n".join("".join(row) for row in canvas)

    # @staticmethod
    # def ascii_dag(adj: dict, labels: dict) -> str:
    #     """
    #     Draw an ASCII representation of a DAG.
    #     - adj: dict[node, list of successor nodes]
    #     - labels: dict[node, str label]
    #     """
    #     # Build reverse adjacency for level assignment
    #     rev = {u: [] for u in adj}
    #     for u, vs in adj.items():
    #         for v in vs:
    #             rev.setdefault(v, []).append(u)
    #
    #     # Compute levels (distance from sources)
    #     levels = {}
    #     def compute_level(u):
    #         if u in levels:
    #             return levels[u]
    #         if not rev.get(u):
    #             levels[u] = 0
    #         else:
    #             levels[u] = max(compute_level(p) for p in rev[u]) + 1
    #         return levels[u]
    #
    #     for u in adj:
    #         compute_level(u)
    #
    #     # Group nodes by level
    #     layers = {}
    #     for u, lvl in levels.items():
    #         layers.setdefault(lvl, []).append(u)
    #     for lvl in layers:
    #         layers[lvl].sort(key=lambda n: str(n))
    #
    #     # Calculate box and spacing sizes
    #     max_label_len = max(len(labels[u]) for u in adj)
    #     box_w = max_label_len + 4  # padding
    #     box_h = 3
    #     h_spacing = 4
    #     v_spacing = 1
    #
    #     # Assign coordinates
    #     coords = {}
    #     for lvl, nodes in layers.items():
    #         for idx, u in enumerate(nodes):
    #             x = lvl * (box_w + h_spacing)
    #             y = idx * (box_h + v_spacing)
    #             coords[u] = (x, y)
    #
    #     # Canvas size
    #     width = max(x + box_w for x, y in coords.values()) + 1
    #     height = max(y + box_h for x, y in coords.values()) + 1
    #
    #     # Initialize canvas
    #     canvas = [[' ' for _ in range(width)] for _ in range(height)]
    #
    #     # Draw boxes
    #     for u, (x, y) in coords.items():
    #         label = labels[u]
    #         top = y
    #         mid = y + 1
    #         bot = y + 2
    #         # Top border
    #         for i in range(box_w):
    #             canvas[top][x + i] = '='
    #             canvas[bot][x + i] = '='
    #         # Middle content
    #         text = f"  {label}  ".center(box_w)
    #         canvas[mid][x] = canvas[mid][x + box_w - 1] = '='
    #         for i, ch in enumerate(text):
    #             canvas[mid][x + 1 + i] = ch
    #
    #     # Draw edges
    #     arrow_margin = 1  # leave this many spaces between box and arrow
    #     for u, vs in adj.items():
    #         x0, y0 = coords[u]
    #         x0r = x0 + box_w
    #         y0m = y0 + 1
    #         for v in vs:
    #             x1, y1 = coords[v]
    #             x1l = x1
    #             y1m = y1 + 1
    #
    #             # Horizontal line, inset by arrow_margin so we don't touch box borders
    #             for x in range(x0r + arrow_margin, x1l - arrow_margin - 1):
    #                 if canvas[y0m][x] == ' ':
    #                     canvas[y0m][x] = '-'
    #
    #             # Arrow head, also inset
    #             arrow_x = x1l - arrow_margin - 1
    #             canvas[y0m][arrow_x] = '>'
    #
    #             # Vertical connector if needed (at the same arrow_x)
    #             if y0m < y1m:
    #                 for yy in range(y0m + 1, y1m):
    #                     if canvas[yy][arrow_x] == ' ':
    #                         canvas[yy][arrow_x] = 'v'
    #             elif y0m > y1m:
    #                 for yy in range(y1m + 1, y0m):
    #                     if canvas[yy][arrow_x] == ' ':
    #                         canvas[yy][arrow_x] = '^'
    #
    #     # Restore mid-line borders for closed boxes
    #     for u, (x, y) in coords.items():
    #         mid = y + 1
    #         canvas[mid][x] = '='
    #         canvas[mid][x + box_w - 1] = '='
    #
    #     return "\n".join("".join(row) for row in canvas)

    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _populate_in_degree_and_id_lookup_dicts_(
        argument: DirectionalAcyclicGraph[T] | DirectionalAcyclicGraphNode[T],
        in_degree: dict[int, int],
        node_lookup: dict[int, DirectionalAcyclicGraphNode[T]],
    ) -> None:

        for node in argument.depth_first_traversal():
            node_id = id(node)
            in_degree[node_id] = len(node.predecessors)
            node_lookup[node_id] = node

    @staticmethod
    def _kahn_algorithm_main_(
        in_degree: dict[int, int],
        zero_in_degree_nodes: list[DirectionalAcyclicGraphNode[T]],
    ) -> Generator[DirectionalAcyclicGraphNode[T], Any, None]:

        while zero_in_degree_nodes:
            current_node = zero_in_degree_nodes.pop()
            yield current_node

            for successor in current_node.successors:
                succ_id = id(successor)
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    zero_in_degree_nodes.append(successor)
