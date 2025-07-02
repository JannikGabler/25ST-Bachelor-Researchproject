from copy import copy
from typing import TypeVar, Generator, Any

from data_structures.directed_acyclic_graph.directional_acyclic_graph import DirectionalAcyclicGraph
from data_structures.directed_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode
from exceptions.not_instantiable_error import NotInstantiableError
from utils.collections_utils import CollectionsUtils

T = TypeVar('T')


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
    def get_entry_nodes_to_node_graph(nodes: list[DirectionalAcyclicGraphNode[T]]) -> list[DirectionalAcyclicGraphNode[T]]:
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
        in_degree: dict[int, int] = {}
        node_lookup: dict[int, DirectionalAcyclicGraphNode[T]] = {}

        DirectionalAcyclicGraphUtils._populate_in_degree_and_id_lookup_dicts_(argument, in_degree, node_lookup)
        zero_in_degree_nodes = [node_lookup[node_id] for node_id, deg in in_degree.items() if deg == 0]

        yield from DirectionalAcyclicGraphUtils._kahn_algorithm_main_(in_degree, zero_in_degree_nodes)



    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _populate_in_degree_and_id_lookup_dicts_(argument: DirectionalAcyclicGraph[T] | DirectionalAcyclicGraphNode[T],
                                                 in_degree: dict[int, int], node_lookup: dict[int, DirectionalAcyclicGraphNode[T]]) -> None:

        for node in argument.death_first_traversal():
            node_id = id(node)
            in_degree[node_id] = len(node.predecessors)
            node_lookup[node_id] = node



    @staticmethod
    def _kahn_algorithm_main_(in_degree: dict[int, int], zero_in_degree_nodes: list[DirectionalAcyclicGraphNode[T]]
                              ) -> Generator[DirectionalAcyclicGraphNode[T], Any, None]:

        while zero_in_degree_nodes:
            current_node = zero_in_degree_nodes.pop()
            yield current_node

            for successor in current_node.successors:
                succ_id = id(successor)
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    zero_in_degree_nodes.append(successor)