class Node:
    def __init__(self, label):
        self.label = label
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return str(self.label)


class Tree:
    def __init__(self, root_label):
        self.root = Node(root_label)

    def add_child(self, parent_node, label):
        child_node = Node(label)
        parent_node.add_child(child_node)
        return child_node

    def __repr__(self):
        return str(self.root)

    def iterate_depth_search(self, node=None):
        if node is None:
            node = self.root
        result = []
        stack = [node]
        while stack:
            current_node = stack.pop()
            result.append(current_node)
            stack.extend(reversed(current_node.children))
        return result


class Forest:
    def __init__(self, trees) -> None:
        self.trees = trees

    def add_tree(self, tree):
        self.trees.append(tree)

    def iterate_trees(self):
        nodes = []
        for tree in self.trees:
            nodes.extend(tree.iterate_depth_search())
        return nodes
