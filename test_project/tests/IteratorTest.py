from test_project.src.Tree import Tree, Forest

def visit_forest():
    tree1 = Tree("A")
    node_b = tree1.add_child(tree1.root, "B")
    node_c = tree1.add_child(tree1.root, "C")
    node_d = tree1.add_child(node_b, "D")
    node_e = tree1.add_child(node_b, "E")
    node_f = tree1.add_child(node_c, "F")

    tree2 = Tree("0")
    node_1 = tree2.add_child(tree2.root, "1")
    node_2 = tree2.add_child(tree2.root, "2")
    node_3 = tree2.add_child(tree2.root, "3")
    node_4 = tree2.add_child(node_2, "4")
    node_5 = tree2.add_child(node_2, "5")

    forest = Forest([tree1, tree2])

    result = []
    for node in forest.iterate_trees():
        print(node)
        result.append(str(node))

    assert result == ['A', 'B', 'D', 'E', 'C', 'F', '0', '1', '2', '4', '5', '3']


if __name__ == '__main__':
    visit_forest()
