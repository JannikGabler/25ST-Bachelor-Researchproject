from __future__ import annotations
from typing import TypeVar, Generic, Any, Generator, Callable, Iterable

from general_data_structures.freezable import Freezable
from general_data_structures.tree.tree_node import TreeNode

T = TypeVar("T")
U = TypeVar("U")


class Tree(Generic[T], Freezable, Iterable[T]):
    ###############################
    ### Attributes of instances ###
    ###############################
    _root_node_: TreeNode[T] | None

    ###################
    ### Constructor ###
    ###################
    def __init__(self, argument: TreeNode[T] | str | None) -> None:
        if argument is not None:
            if isinstance(argument, TreeNode):
                self._init_from_root_node_(argument)
            elif isinstance(argument, str):
                self._init_from_str_(argument)
            else:
                raise TypeError(
                    f"Got argument of type '{type(argument)}' but must be a TreeNode[T], str or None."
                )
        else:
            self._root_node_ = None

    ######################
    ### Public methods ###
    ######################
    def pre_order_traversal(self) -> Generator[TreeNode[T], Any, None]:
        if self._root_node_:
            yield from self._root_node_.pre_order_traversal()

    def post_order_traversal(self) -> Generator[TreeNode[T], Any, None]:
        if self._root_node_:
            yield from self._root_node_.post_order_traversal()

    def value_map(self, value_mapping: Callable[[T], U]) -> Tree[U]:
        if self._root_node_ is None:
            return Tree[U](None)
        else:
            new_root_node: TreeNode[U] = self._root_node_.value_map(value_mapping)
            return Tree[U](new_root_node)

    #########################
    ### Getters & setters ###
    #########################
    @property
    def root_node(self) -> TreeNode[T] | None:
        return self._root_node_

    @property
    def amount_of_nodes(self) -> int:
        result: int = 0

        for _ in self.pre_order_traversal():
            result += 1

        return result

    ##########################
    ### Overridden methods ###
    ##########################
    def __repr__(self) -> str:
        return f"Tree(root_node={repr(self._root_node_)})"

    def __iter__(self) -> Generator[TreeNode[T], Any, None]:
        yield from self.pre_order_traversal()

    def __eq__(self, other: any) -> bool:
        if not isinstance(other, Tree):  # Checks for None
            return False
        else:
            own_gen = self.pre_order_traversal()
            other_gen = other.pre_order_traversal()

            while True:
                own_next = next(own_gen, None)
                other_next = next(other_gen, None)

                if own_next is None and other_next is None:
                    return True
                elif own_next != other_next:
                    return False

    def __hash__(self):
        if self._frozen_:  # instance is immutable
            hash_values: list[int] = [node.__hash__() for node in self]
            return hash(hash_values)
        else:  # instance is mutable
            return None

    def freeze(self) -> None:
        super().freeze()

        for child in self.pre_order_traversal():
            child.freeze()

    #######################
    ### Private methods ###
    #######################
    def _init_from_root_node_(self, root_node: TreeNode[T]) -> None:
        self._root_node_ = root_node

    def _init_from_str_(self, string: str) -> None:
        lines: list[str] = string.strip("\n").splitlines()
        if not lines:
            self._root_node_ = None
            return

        stack: list[tuple[int, TreeNode[str]]] = []

        for line in lines:
            stripped: str = line.lstrip()

            if stripped:
                indent: int = len(line) - len(stripped)
                node: TreeNode[str] = TreeNode(stripped)

                while stack and indent <= stack[-1][0]:
                    stack.pop()

                if not stack:
                    self._root_node_ = node
                else:
                    parent_node = stack[-1][1]
                    parent_node.add_child_node(node)

                stack.append((indent, node))
