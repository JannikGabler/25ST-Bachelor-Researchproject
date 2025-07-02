from copy import copy
from typing import TypeVar

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