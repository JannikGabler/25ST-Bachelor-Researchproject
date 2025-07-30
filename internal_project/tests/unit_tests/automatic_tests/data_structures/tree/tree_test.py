import unittest

from general_data_structures.tree.tree import Tree
from general_data_structures.tree.tree_node import TreeNode


class MyTestCase(unittest.TestCase):
    def test_creation_from_correct_string(self):
        tree_str = """
        root
          child1
            child1.1
            child1.2
          child2
            child2.1
        \tchild3
        \t  child3.1
        """

        root_node: TreeNode[str] = TreeNode("root")
        child_1_node: TreeNode[str] = TreeNode("child1")
        child_1_1_node: TreeNode[str] = TreeNode("child1.1")
        child_1_2_node: TreeNode[str] = TreeNode("child1.2")
        child_2_node: TreeNode[str] = TreeNode("child2")
        child_2_1_node: TreeNode[str] = TreeNode("child2.1")
        child_3_node: TreeNode[str] = TreeNode("child3")
        child_3_1_node: TreeNode[str] = TreeNode("child3.1")

        root_node.add_child_nodes([child_1_node, child_2_node, child_3_node])
        child_1_node.add_child_nodes([child_1_1_node, child_1_2_node])
        child_2_node.add_child_node(child_2_1_node)
        child_3_node.add_child_node(child_3_1_node)

        expected: Tree[str] = Tree(root_node)
        result: Tree[str] = Tree(tree_str)

        self.assertEqual(expected, result)

if __name__ == '__main__':
    unittest.main()
