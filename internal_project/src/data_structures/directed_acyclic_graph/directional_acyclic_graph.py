from __future__ import annotations
from collections.abc import Iterable
from copy import copy
from itertools import zip_longest
from typing import TypeVar, Generic, Generator, Any, Callable

from data_structures.directed_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode
from data_structures.freezable import Freezable
from exceptions.duplicate_value_error import DuplicateError
from exceptions.invalid_argument_exception import InvalidArgumentException
from utils.collections_utils import CollectionsUtils
from utils.directional_acyclic_graph_utils import DirectionalAcyclicGraphUtils
from utils.ini_format_utils import INIFormatUtils

T = TypeVar('T')
U = TypeVar('U')

class DirectionalAcyclicGraph(Generic[T], Freezable):
    ###############################
    ### Attributes of instances ###
    ###############################
    _entry_nodes_: list[DirectionalAcyclicGraphNode[T]]



    ###################
    ### Constructor ###
    ###################
    def __init__(self, argument: DirectionalAcyclicGraphNode[T] | list[DirectionalAcyclicGraphNode[T]] | str | None = None) -> None:
        self._entry_nodes_ = []

        if argument is not None:
            if isinstance(argument, DirectionalAcyclicGraphNode):
                self._init_from_nodes_([argument])
            elif isinstance(argument, list):
                self._init_from_nodes_(argument)
            elif isinstance(argument, str):
                self._init_from_str_(argument)
            else:
                raise TypeError(f"Got argument of type '{type(argument)}' but must be a DirectionalAcyclicGraphNode[T], "
                                f"list[DirectionalAcyclicGraphNode[T]], str or None.")



    ######################
    ### Public methods ###
    ######################
    def add_node(self, node: DirectionalAcyclicGraphNode[T]) -> None:
        if self.contains_node(node):
            raise DuplicateError(f"Cannot add '{str(node)}' to graph because it is already contained in the graph.")

        new_entry_nodes: list[DirectionalAcyclicGraphNode[T]] = DirectionalAcyclicGraphUtils.get_entry_nodes_to_node_graph([node])
        self._entry_nodes_.extend(new_entry_nodes)



    def add_node_from_value(self, value: T) -> None:
        node: DirectionalAcyclicGraphNode[T] = DirectionalAcyclicGraphNode(value)
        self.add_node(node)



    def add_edge(self, orig_node: DirectionalAcyclicGraphNode[T], dest_node: DirectionalAcyclicGraphNode[T]) -> None:
        if not self.contains_node(orig_node):
            raise InvalidArgumentException(f"Cannot add an edge from the node '{str(orig_node)}' to the node '{str(dest_node)}' because the origin node is not contained in the graph.'")

        orig_node.add_successor(dest_node)
        self._entry_nodes_ = DirectionalAcyclicGraphUtils.get_entry_nodes_to_node_graph(self._entry_nodes_)



    def remove_node(self, node: DirectionalAcyclicGraphNode[T]) -> None:
        if not self.contains_node(node):
            raise InvalidArgumentException(f"Cannot remove the node '{str(node)}' from the graph because it is not contained in the graph.")
        if CollectionsUtils.is_exact_element_in_collection(node, self._entry_nodes_):
            self._entry_nodes_.remove(node)

        potential_new_entry_nodes: list[DirectionalAcyclicGraphNode[T]] = self._entry_nodes_ + node.successors

        node.clear_predecessors()
        node.clear_successors()

        self._entry_nodes_ = DirectionalAcyclicGraphUtils.get_entry_nodes_to_node_graph(potential_new_entry_nodes)



    def remove_edge(self, orig_node: DirectionalAcyclicGraphNode[T], dest_node: DirectionalAcyclicGraphNode[T]) -> None:
        if not self.contains_node(orig_node):
            raise InvalidArgumentException(f"Cannot remove the edge from the node '{str(orig_node)}' to the node '{str(dest_node)}' because the edge is not contained in the graph.'")

        potential_new_entry_nodes: list[DirectionalAcyclicGraphNode[T]] = self._entry_nodes_ + [dest_node]
        self._entry_nodes_ = DirectionalAcyclicGraphUtils.get_entry_nodes_to_node_graph(potential_new_entry_nodes)



    def contains_node(self, node: DirectionalAcyclicGraphNode[T]) -> bool:
        for entry_node in self._entry_nodes_:
            if entry_node.can_reach_node(node):
                return True

        return False



    def death_first_traversal(self) -> Generator[DirectionalAcyclicGraphNode[T], Any, None]:
        stack: list[DirectionalAcyclicGraphNode[T]] = copy(self._entry_nodes_)
        traversed_nodes: set[int] = set()

        while stack:
            current_node: DirectionalAcyclicGraphNode[T] = stack.pop()

            stack.extend(current_node.successors)

            if id(current_node) not in traversed_nodes:
                traversed_nodes.add(id(current_node))
                yield current_node



    def breadth_first_traversal(self) -> Generator[DirectionalAcyclicGraphNode[T], Any, None]:
        queue: list[DirectionalAcyclicGraphNode[T]] = copy(self._entry_nodes_)
        traversed_nodes: set[int] = set()

        while queue:
            current_node: DirectionalAcyclicGraphNode[T] = queue.pop(0)

            queue.extend(current_node.successors)

            if id(current_node) not in traversed_nodes:
                traversed_nodes.add(id(current_node))
                yield current_node



    def value_map(self, value_mapping: Callable[[T], U]) -> DirectionalAcyclicGraph[U]:
        if not self._entry_nodes_:
            return DirectionalAcyclicGraph[U]()
        else:
            return self._create_mapped_dag_based_on_self_(value_mapping)



    #########################
    ### Getters & setters ###
    #########################
    @property
    def entry_nodes(self) -> list[DirectionalAcyclicGraphNode[T]]:
        return copy(self._entry_nodes_)



    ##########################
    ### Overridden methods ###
    ##########################
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DirectionalAcyclicGraph):
            return False

        for own_node, other_node in zip_longest(self.breadth_first_traversal(), other.breadth_first_traversal()):
            if own_node != other_node:
                return False

        return True


    def __hash__(self) -> int | None:
        if self._frozen_:   # instance is immutable
            hash_values: list[int] = []

            for node in self.breadth_first_traversal():
                hash_values.append(hash(node))

            return hash(hash_values)

        else:   # instance is mutable
            return None


    def __iter__(self) -> Generator[DirectionalAcyclicGraphNode[T], Any, None]:
        yield from self.breadth_first_traversal()



    #######################
    ### Private methods ###
    #######################
    def _init_from_nodes_(self, nodes : list[DirectionalAcyclicGraphNode[T]]) -> None:
        self._entry_nodes_ = DirectionalAcyclicGraphUtils.get_entry_nodes_to_node_graph(nodes)



    def _init_from_str_(self, string: str) -> None:
        entry_list: list[str] = INIFormatUtils.split_into_entries(string)
        entry_dict: dict[str, str] = INIFormatUtils.split_entries_into_key_value_pairs(entry_list)
        nodes, edges = DirectionalAcyclicGraph._convert_key_value_pairs_into_nodes_and_edges_in_string_init_(entry_dict)
        self._construct_graph_from_nodes_and_edges_in_string_init_(nodes, edges)



    def _construct_graph_from_nodes_and_edges_in_string_init_(self, nodes: dict[str, DirectionalAcyclicGraphNode[tuple[str, str, dict[str, str]]]], edges: list[tuple[str, str]]) -> None:
        for node in nodes.values():
            self.add_node(node)

        for edge in edges:
            if edge[0] not in nodes:
                raise TypeError(f"Cannot parse the string into an directional acyclic graph."
                                f"There is no node with the id '{edge[0]}' but it's specified as the origin of the edge '{edge}'.")
            if edge[1] not in nodes:
                raise TypeError(f"Cannot parse the string into an directional acyclic graph."
                                f"There is no node with the id '{edge[1]}' but it's specified as the destination of the edge '{edge}'.")

            source_node: DirectionalAcyclicGraphNode[tuple[str, str, dict[str, str]]] = nodes[edge[0]]
            dest_node: DirectionalAcyclicGraphNode[tuple[str, str, dict[str, str]]] = nodes[edge[1]]
            self.add_edge(source_node, dest_node)



    @staticmethod
    def _convert_key_value_pairs_into_nodes_and_edges_in_string_init_(entry_dict: dict[str, str]) -> tuple[dict[str, DirectionalAcyclicGraphNode[tuple[str, str, dict[str, str]]]], list[tuple[str, str]]]:
        nodes: dict[str, DirectionalAcyclicGraphNode[T]] = {}
        edges: list[tuple[str, str]] = []

        for key, value in entry_dict.items():
            DirectionalAcyclicGraph._handle_node_key_value_pair_in_string_init_(key, value, nodes, edges)

        return nodes, edges



    @staticmethod
    def _handle_node_key_value_pair_in_string_init_(key: str, value: str, nodes: dict[str, DirectionalAcyclicGraphNode[T]], edges: list[tuple[str, str]]) -> None:
        value_lines: list[str] = value.splitlines()
        node_id: str = key
        node_type: str = value_lines[0]
        properties: dict[str, str] = {}

        if len(value_lines) >= 2:
            property_entries: list[str] = INIFormatUtils.split_into_entries('\n'.join(value_lines[1:]))
            properties = INIFormatUtils.split_entries_into_key_value_pairs(property_entries)

        DirectionalAcyclicGraph._add_edges_from_node_properties_in_string_init_(node_id, properties, edges)

        node_value: tuple[str, str, dict[str, str]] = (node_id, node_type, properties)
        new_node: DirectionalAcyclicGraphNode[tuple[str, str, dict[str, str]]] = DirectionalAcyclicGraphNode(node_value)
        nodes[node_id] = new_node



    @staticmethod
    def _add_edges_from_node_properties_in_string_init_(node_id: str, node_properties: dict[str, str], edges: list[tuple[str, str]]) -> None:
        DirectionalAcyclicGraph._add_predecessors_from_node_properties_in_string_init_(node_id, node_properties, edges)
        DirectionalAcyclicGraph._add_successors_from_node_properties_in_string_init_(node_id, node_properties, edges)



    @staticmethod
    def _add_predecessors_from_node_properties_in_string_init_(node_id: str, node_properties: dict[str, str], edges: list[tuple[str, str]]) -> None:
        if "predecessors" in node_properties:
            predecessors_string: str = node_properties["predecessors"]
            predecessors_object: Any = eval(predecessors_string)

            if not isinstance(predecessors_object, Iterable):
                raise TypeError(f"Cannot parse the string into an directional acyclic graph."
                                f"The 'predecessors' field of the node '{node_id}' must be iterable but is of type"
                                f"'{type(predecessors_object)}',which is not iterable.")

            for predecessor in predecessors_object:
                if not isinstance(predecessor, str):
                    raise TypeError(f"Cannot parse the string into an directional acyclic graph."
                                    f"The item '{repr(predecessor)}' of the 'predecessors' field from the node"
                                    f"'{node_id}' must be a string but is from type '{type(predecessor)}'.")

                edges.append((predecessor, node_id))

            del node_properties["predecessors"]


    @staticmethod
    def _add_successors_from_node_properties_in_string_init_(node_id: str, node_properties: dict[str, str], edges: list[tuple[str, str]]) -> None:
        if "successors" in node_properties:
            successors_string: str = node_properties["successors"]
            successors_object: Any = eval(successors_string)

            if not isinstance(successors_object, Iterable):
                raise TypeError(f"Cannot parse the string into an directional acyclic graph."
                                f"The 'successors' field of the node '{node_id}' must be iterable but is of type"
                                f"'{type(successors_object)}',which is not iterable.")

            for successor in successors_object:
                if not isinstance(successor, str):
                    raise TypeError(f"Cannot parse the string into an directional acyclic graph."
                                    f"The item '{repr(successor)}' of the 'predecessors' field from the node"
                                    f"'{node_id}' must be a string but is from type '{type(successor)}'.")

                edges.append((node_id, successor))

            del node_properties["successors"]



    def _create_mapped_dag_based_on_self_(self, value_mapping: Callable[[T], U]) -> DirectionalAcyclicGraph[U]:
        node_mapping: dict[int, DirectionalAcyclicGraphNode[U]] = {}
        node_queue: list[DirectionalAcyclicGraphNode[T]] = []
        new_entry_nodes: list[DirectionalAcyclicGraphNode[U]] = []

        for old_node in self._entry_nodes_:
            new_node = DirectionalAcyclicGraph._create_mapped_new_node_(old_node, value_mapping, node_mapping, node_queue)
            new_entry_nodes.append(new_node)

        while node_queue:
            old_node: DirectionalAcyclicGraphNode[T] = node_queue.pop(0)

            for successor in old_node.successors:
                if id(successor) not in node_mapping:
                    DirectionalAcyclicGraph._create_mapped_new_node_(successor, value_mapping, node_mapping, node_queue)

                node_mapping[id(old_node)].add_successor(node_mapping[id(successor)])

        return DirectionalAcyclicGraph[U](new_entry_nodes)



    @staticmethod
    def _create_mapped_new_node_(old_node: DirectionalAcyclicGraphNode[T], value_mapping: Callable[[T], U],
                                 node_mapping: dict[int, DirectionalAcyclicGraphNode[U]],
                                 node_queue: list[DirectionalAcyclicGraphNode[T]]) -> DirectionalAcyclicGraphNode[U]:

        new_value: U = value_mapping(old_node.value)
        new_node: DirectionalAcyclicGraphNode[U] = DirectionalAcyclicGraphNode[U](new_value)
        node_mapping[id(old_node)] = new_node
        node_queue.append(old_node)

        return new_node





