from __future__ import annotations

from copy import copy
from typing import Generic, TypeVar, Any, Generator

from data_structures.freezable import Freezable
from exceptions.cycle_exception import CycleException
from exceptions.duplicate_value_error import DuplicateError
from utils.collections_utils import CollectionsUtils

T = TypeVar('T')

class DirectionalAcyclicGraphNode(Generic[T], Freezable):
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
        self._predecessors_ = []
        self._successors_ = []
        self._value_ = value



    ######################
    ### Public methods ###
    ######################
    def add_predecessor(self, predecessor: DirectionalAcyclicGraphNode[T]) -> None:
        if DirectionalAcyclicGraphNode._is_predecessor_of_node_(predecessor, self):
            raise DuplicateError(f"Cannot add '{str(predecessor)}' as a predecessor because this node is already a predecessor.")
        if self.can_reach_node(predecessor):
            raise CycleException(f"Cannot add '{str(predecessor)}' as a predecessor because it would create a cycle.")

        predecessor._successors_.append(self)
        self._predecessors_.append(predecessor)

    def remove_predecessor(self, predecessor: DirectionalAcyclicGraphNode[T]) -> None:
        if DirectionalAcyclicGraphNode._is_predecessor_of_node_(predecessor, self):
            self._predecessors_.remove(predecessor)
            predecessor._successors_.remove(self)
        else:
            raise Exception(f"Cannot remove '{str(predecessor)}' from predecessors because the node is not a predecessor.")

    def clear_predecessors(self) -> None:
        for predecessor in self._predecessors_:
            self.remove_predecessor(predecessor)

    def is_predecessor_of(self, node: DirectionalAcyclicGraphNode[T]) -> bool:
        return CollectionsUtils.is_exact_element_in_collection(node, self._successors_)



    def add_successor(self, successor: DirectionalAcyclicGraphNode[T]) -> None:
        if DirectionalAcyclicGraphNode._is_successor_of_node_(successor, self):
            raise DuplicateError(f"Cannot add '{str(successor)}' as a successor because this node is already a successor.")
        if successor.can_reach_node(self):
            raise CycleException(f"Cannot add '{str(successor)}' as a successor because it would create a cycle.")

        self._successors_.append(successor)
        successor._predecessors_.append(self)

    def remove_successor(self, successor: DirectionalAcyclicGraphNode[T]) -> None:
        if DirectionalAcyclicGraphNode._is_successor_of_node_(successor, self):
            self._successors_.remove(successor)
            successor._predecessors_.remove(self)
        else:
            raise Exception(f"Cannot remove '{str(successor)}' from successors because the node is not a successor.")

    def clear_successors(self) -> None:
        for successor in self._successors_:
            self.remove_successor(successor)

    def is_successor_of(self, node: DirectionalAcyclicGraphNode[T]) -> bool:
        return CollectionsUtils.is_exact_element_in_collection(node, self._predecessors_)



    def death_first_traversal(self) -> Generator[DirectionalAcyclicGraphNode[T], Any, None]:
        stack: list[DirectionalAcyclicGraphNode[T]] = [self]

        while stack:
            current_node: DirectionalAcyclicGraphNode[T] = stack.pop()

            stack.extend(current_node._successors_)

            yield current_node

    def breadth_first_traversal(self) -> Generator[DirectionalAcyclicGraphNode[T], Any, None]:
        queue: list[DirectionalAcyclicGraphNode[T]] = [self]

        while queue:
            current_node: DirectionalAcyclicGraphNode[T] = queue.pop(0)

            queue.extend(current_node._successors_)

            yield current_node



    def can_reach_node(self, node: DirectionalAcyclicGraphNode[T]) -> bool:
        for reachable_node in self.breadth_first_traversal():
            if reachable_node is node:
                return True

        return False



    #########################
    ### Getters & setters ###
    #########################
    @property
    def predecessors(self) -> list[DirectionalAcyclicGraphNode[T]]:
        return copy(self._predecessors_)

    @property
    def successors(self) -> list[DirectionalAcyclicGraphNode[T]]:
        return copy(self._successors_)

    @property
    def value(self) -> T | None:
        return self._value_



    ##########################
    ### Overridden methods ###
    ##########################
    def __eq__(self, other: any) -> bool:
        if not isinstance(other, DirectionalAcyclicGraphNode): # Cover None
            return False
        if self.value != other.value:
            return False
        if len(self._predecessors_) != len(other._predecessors_):
            return False
        if len(self._successors_) != len(other._successors_):
            return False

        return True



    def __hash__(self) -> int | None:
        if self._frozen_:   # instance is immutable
            return hash((self._value_, len(self._predecessors_), len(self._successors_)))

        else:   # instance is mutable
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