from __future__ import annotations

from collections import deque
from copy import copy
from typing import Generic, TypeVar, Any, Generator

from general_data_structures.freezable import Freezable
from exceptions.cycle_exception import CycleException
from exceptions.duplicate_value_error import DuplicateError
from utils.collections_utils import CollectionsUtils
from utils.directional_acyclic_graph_utils import DirectionalAcyclicGraphUtils

T = TypeVar("T")


class DirectionalAcyclicGraphNode(Generic[T], Freezable):
    """
    Node in a directed acyclic graph (DAG) holding an optional value and references to predecessor and successor nodes.
    """


    ###############################
    ### Attributes of instances ###
    ###############################
    _predecessors_: list[DirectionalAcyclicGraphNode[T]]
    _successors_: list[DirectionalAcyclicGraphNode[T]]
    _value_: T | None


    ###################
    ### Constructor ###
    ###################
    def __init__(self, value: T | None = None):
        """
        Args:
            value: Optional payload stored in the node.
        """

        self._predecessors_ = []
        self._successors_ = []
        self._value_ = value


    ######################
    ### Public methods ###
    ######################
    def add_predecessor(self, predecessor: DirectionalAcyclicGraphNode[T]) -> None:
        """
        Add the given node as a predecessor of this node.

        Args:
            predecessor (DirectionalAcyclicGraphNode[T]): Node to add as predecessor.

        Returns:
            None

        Raises:
            DuplicateError: If the node is already a predecessor.
            CycleException: If adding the edge would create a cycle.
        """

        if DirectionalAcyclicGraphNode._is_predecessor_of_node_(predecessor, self):
            raise DuplicateError(f"Cannot add '{str(predecessor)}' as a predecessor because this node is already a predecessor.")
        if self.can_reach_node(predecessor):
            raise CycleException(f"Cannot add '{str(predecessor)}' as a predecessor because it would create a cycle.")

        predecessor._successors_.append(self)
        self._predecessors_.append(predecessor)


    def remove_predecessor(self, predecessor: DirectionalAcyclicGraphNode[T]) -> None:
        """
        Remove the given node from the predecessors of this node.

        Args:
            predecessor (DirectionalAcyclicGraphNode[T]): Node to remove.

        Returns:
            None

        Raises:
            Exception: If the node is not a predecessor of this node.
        """

        if DirectionalAcyclicGraphNode._is_predecessor_of_node_(predecessor, self):
            self._predecessors_.remove(predecessor)
            predecessor._successors_.remove(self)
        else:
            raise Exception(f"Cannot remove '{str(predecessor)}' from predecessors because the node is not a predecessor.")


    def clear_predecessors(self) -> None:
        """
        Remove all predecessors from this node.

        Returns:
            None
        """

        for predecessor in self._predecessors_:
            self.remove_predecessor(predecessor)


    def is_predecessor_of(self, node: DirectionalAcyclicGraphNode[T]) -> bool:
        """
        Check whether this node is a direct predecessor of the given node.

        Args:
            node (DirectionalAcyclicGraphNode[T]): Node to check against.

        Returns:
             bool: True if this node is a predecessor of the given node,
                   False otherwise.
        """

        return CollectionsUtils.is_exact_element_in_collection(node, self._successors_)


    def add_successor(self, successor: DirectionalAcyclicGraphNode[T]) -> None:
        """
        Add the given node as a successor of this node.

        Args:
            successor (DirectionalAcyclicGraphNode[T]): Node to add as successor.

        Raises:
            DuplicateError: If the node is already a successor.
            CycleException: If adding the node would create a cycle.
        """

        if DirectionalAcyclicGraphNode._is_successor_of_node_(successor, self):
            raise DuplicateError(f"Cannot add '{str(successor)}' as a successor because this node is already a successor.")
        if successor.can_reach_node(self):
            raise CycleException(f"Cannot add '{str(successor)}' as a successor because it would create a cycle.")

        self._successors_.append(successor)
        successor._predecessors_.append(self)


    def remove_successor(self, successor: DirectionalAcyclicGraphNode[T]) -> None:
        """
        Remove the given node from the successors of this node.

        Args:
            successor (DirectionalAcyclicGraphNode[T]): Node to remove.

        Raises:
            Exception: If the node is not a successor of this node.
        """

        if DirectionalAcyclicGraphNode._is_successor_of_node_(successor, self):
            self._successors_.remove(successor)
            successor._predecessors_.remove(self)
        else:
            raise Exception(f"Cannot remove '{str(successor)}' from successors because the node is not a successor.")


    def clear_successors(self) -> None:
        """
        Remove all successors from this node.

        Returns:
            None
        """

        for successor in self._successors_:
            self.remove_successor(successor)


    def is_successor_of(self, node: DirectionalAcyclicGraphNode[T]) -> bool:
        """
        Check whether this node is a direct successor of the given node.

        Args:
            node (DirectionalAcyclicGraphNode[T]): Node to check against.

        Returns:
            bool: True if this node is a successor of the given node,
                  False otherwise.
        """

        return CollectionsUtils.is_exact_element_in_collection(node, self._predecessors_)


    def depth_first_traversal(self) -> Generator[DirectionalAcyclicGraphNode[T], Any, None]:
        """
        Iterate nodes in depth-first order starting from this node.

        Returns:
            Generator[DirectionalAcyclicGraphNode[T], Any, None]: Generator over visited nodes.
        """

        stack: deque[DirectionalAcyclicGraphNode[T]] = deque([self])

        while stack:
            current_node: DirectionalAcyclicGraphNode[T] = stack.pop()

            stack.extend(current_node._successors_)

            yield current_node


    def breadth_first_traversal(self) -> Generator[DirectionalAcyclicGraphNode[T], Any, None]:
        """
        Iterate nodes in breadth-first order starting from this node.

        Returns:
            Generator[DirectionalAcyclicGraphNode[T], Any, None]: Generator over visited nodes.
        """

        queue: deque[DirectionalAcyclicGraphNode[T]] = deque([self])

        while queue:
            current_node: DirectionalAcyclicGraphNode[T] = queue.popleft()

            queue.extend(current_node._successors_)

            yield current_node


    def topological_traversal(self) -> Generator[DirectionalAcyclicGraphNode[T], Any, None]:
        """
        Iterate nodes in topological order starting from this node.

        Returns:
            Generator[DirectionalAcyclicGraphNode[T], Any, None]: Generator over nodes in topological order.
        """

        yield from DirectionalAcyclicGraphUtils.topological_traversal(self)


    def all_predecessors_traversal(self) -> Generator[DirectionalAcyclicGraphNode[T], Any, None]:
        """
        Iterate all transitive predecessors of this node.

        Returns:
            Generator[DirectionalAcyclicGraphNode[T], Any, None]: Generator over predecessor nodes.
        """

        queue: deque[DirectionalAcyclicGraphNode[T]] = deque(self._predecessors_)

        while queue:
            predecessor: DirectionalAcyclicGraphNode[T] = queue.popleft()

            queue.extend(predecessor._predecessors_)

            yield predecessor


    def can_reach_node(self, node: DirectionalAcyclicGraphNode[T]) -> bool:
        """
        Check whether this node can reach the given node.

        Args:
            node (DirectionalAcyclicGraphNode[T]): Target node.

        Returns:
            bool: True if the given node is reachable from this node,
                  False otherwise.
        """

        for reachable_node in self.breadth_first_traversal():
            if reachable_node is node:
                return True

        return False


    #########################
    ### Getters & setters ###
    #########################
    @property
    def predecessors(self) -> list[DirectionalAcyclicGraphNode[T]]:
        """
        Return a copy of the direct predecessors of this node.

        Returns:
            list[DirectionalAcyclicGraphNode[T]]: List of predecessor nodes.
        """

        return copy(self._predecessors_)


    @property
    def successors(self) -> list[DirectionalAcyclicGraphNode[T]]:
        """
        Return a copy of the direct successors of this node.

        Returns:
            list[DirectionalAcyclicGraphNode[T]]: List of successor nodes.
        """

        return copy(self._successors_)


    @property
    def value(self) -> T | None:
        """
        Return the value stored in this node.

        Returns:
            T | None: The payload value if set, otherwise None.
        """

        return self._value_


    ##########################
    ### Overridden methods ###
    ##########################
    def __eq__(self, other: any) -> bool:
        if not isinstance(other, DirectionalAcyclicGraphNode):  # Cover None
            return False
        if self.value != other.value:
            return False
        if len(self._predecessors_) != len(other._predecessors_):
            return False
        if len(self._successors_) != len(other._successors_):
            return False

        return True


    def __hash__(self) -> int | None:
        if self._frozen_:  # instance is immutable
            return hash(
                (self._value_, len(self._predecessors_), len(self._successors_))
            )

        else:  # instance is mutable
            return None


    def __str__(self) -> str:
        return f"DirectedAcyclicNode(value='{self._value_})'"


    def __repr__(self) -> str:
        return f"DirectedAcyclicNode(value={repr(self._value_)}, predecessors={repr(self._predecessors_)}, successors={repr(self._successors_)}"


    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _is_predecessor_of_node_(predecessor: DirectionalAcyclicGraphNode[T], node: DirectionalAcyclicGraphNode[T]) -> bool:
        for old_predecessor in node._predecessors_:
            if old_predecessor is predecessor:
                return True

        return False


    @staticmethod
    def _is_successor_of_node_(successor: DirectionalAcyclicGraphNode[T], node: DirectionalAcyclicGraphNode[T]) -> bool:
        for old_successor in node._successors_:
            if old_successor is successor:
                return True

        return False
