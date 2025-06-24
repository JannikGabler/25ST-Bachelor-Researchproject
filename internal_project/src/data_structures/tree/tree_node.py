from __future__ import annotations

from copy import copy
from typing import TypeVar, Generic, Any, Generator, Callable

from data_structures.freezable import Freezable

T = TypeVar('T')
U = TypeVar('U')


class TreeNode(Generic[T], Freezable):
    ###############################
    ### Attributes of instances ###
    ###############################
    _parent_node_: TreeNode[T] | None
    _child_nodes_: list[TreeNode[T]]
    _value_: T | None



    ###################
    ### Constructor ###
    ###################
    def __init__(self, value: T | None = None):
        self._parent_node_ = None
        self._child_nodes_ = []
        self._value_ = value



    ######################
    ### Public methods ###
    ######################
    def add_child_node(self, new_child_node: TreeNode[T]) -> None:
        for node in new_child_node.pre_order_traversal():
            if node is self:
                raise Exception(f"Cannot add '{str(new_child_node)}' as a child node because this would create a cycle in the tree.")

        new_child_node.__change_parent_node__(self)



    def add_child_nodes(self, new_child_nodes: set[TreeNode[T]] | list[TreeNode[T]]) -> None:
        for node in new_child_nodes:
            self.add_child_node(node)



    def remove_child_node(self, child_node: TreeNode[T]) -> None:
        if child_node not in self._child_nodes_:
           raise Exception(f"Cannot remove '{str(child_node)}' from child nodes because the node is not a child node.")

        child_node.__change_parent_node__(None)



    def pre_order_traversal(self) -> Generator[TreeNode[T], Any, None]:
        yield self

        for child_node in self._child_nodes_:
            for node in child_node.pre_order_traversal():
                yield node



    def post_order_traversal(self) -> Generator[TreeNode[T], Any, None]:
        for child_node in self._child_nodes_:
            for node in child_node.post_order_traversal():
                yield node

        yield self



    def value_map(self, value_mapping: Callable[[T], U]) -> TreeNode[U]:
        new_children: list[TreeNode[U]] = []

        for child in self._child_nodes_:
            new_child: TreeNode[U] = child.value_map(value_mapping)
            new_children.append(new_child)

        new_value: U = value_mapping(self.value)
        new_node: TreeNode[U] = TreeNode[U](new_value)
        new_node.add_child_nodes(new_children)
        return new_node



    #########################
    ### Getters & setters ###
    #########################
    @property
    def parent_node(self) -> TreeNode[T] | None:
        return self._parent_node_


    @property
    def child_nodes(self) -> list[TreeNode[T]]:
        return copy(self._child_nodes_)


    @property
    def value(self) -> T | None:
        return self._value_


    @property
    def path(self) -> str:
        if self._parent_node_:
            parents_path: str = self._parent_node_.path
            return f"{parents_path}{self._parent_node_._child_nodes_.index(self)}/"
        else:
            return "/"



    ##########################
    ### Overridden methods ###
    ##########################
    def __eq__(self, other: any) -> bool:
        if not isinstance(other, TreeNode): # Cover None
            return False
        if self.value != other.value:
            return False
        if len(self._child_nodes_) != len(other._child_nodes_):
            return False

        return (self.parent_node and other.parent_node) or (self.parent_node is None and other.parent_node is None)



    def __hash__(self) -> int | None:
        if self.__frozen__:   # instance is immutable
            parent_node_value: int = 1 if self._parent_node_ else 0

            return hash((self._value_, parent_node_value, len(self._child_nodes_)))

        else:   # instance is mutable
            return None



    def __repr__(self) -> str:
        return f"TreeNode(value={repr(self._value_)}, child_nodes={repr(self._child_nodes_)})"



    def __str__(self) -> str:
        return f"TreeNode(value='{self._value_})'"



    #######################
    ### Private methods ###
    #######################
    def __change_parent_node__(self, new_parent_node: TreeNode[T] | None) -> None:
        if self._parent_node_:
            self._parent_node_._child_nodes_.remove(self)

        if new_parent_node:
            new_parent_node._child_nodes_.append(self)

        self._parent_node_ = new_parent_node





