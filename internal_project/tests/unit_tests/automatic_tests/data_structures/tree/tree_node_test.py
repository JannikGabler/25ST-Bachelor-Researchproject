import unittest

from general_data_structures.tree.tree_node import TreeNode


class MyTestCase(unittest.TestCase):
    def test_tree_node_creation(self):
        node_1: TreeNode[str] = TreeNode("node_1")
        node_2: TreeNode[int] = TreeNode(7)

        self.assertEqual("node_1", node_1.value)
        self.assertEqual(7, node_2.value)


    def test_node_linking(self):
        node_1: TreeNode[str] = TreeNode("node_1")
        node_1_1: TreeNode[str] = TreeNode("node_1_1")
        node_1_2: TreeNode[str] = TreeNode("node_1_2")
        node_1.add_child_nodes([node_1_1, node_1_2])


        self.assertEqual(None, node_1.parent_node)
        self.assertEqual({node_1_1, node_1_2}, node_1.child_nodes)

        self.assertEqual(node_1, node_1_1.parent_node)
        self.assertEqual(set(), node_1_1.child_nodes)

        self.assertEqual(node_1, node_1_2.parent_node)
        self.assertEqual(set(), node_1_2.child_nodes)


if __name__ == '__main__':
    unittest.main()
