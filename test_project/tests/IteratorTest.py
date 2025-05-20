from test_project.src.Tree import Tree, Forest

def test_visit_forest():
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
        result.append(str(node))

    assert result == ['A', 'B', 'D', 'E', 'C', 'F', '0', '1', '2', '4', '5', '3']

def test_empty_forest():
    forest = Forest([])
    result = list(forest.iterate_trees())
    assert result == []

def test_single_node_tree():
    tree = Tree("A")
    forest = Forest([tree])
    result = forest.iterate_trees()
    nodes = []
    for node in result:
        nodes.append(str(node))
    assert nodes == ["A"]

def test_identical_trees():
    tree1 = Tree("Root")
    node_a = tree1.add_child(tree1.root, "A")
    node_b = tree1.add_child(tree1.root, "B")

    tree2 = Tree("Root")
    node_a2 = tree2.add_child(tree2.root, "A")
    node_b2 = tree2.add_child(tree2.root, "B")

    forest = Forest([tree1, tree2])
    nodes = forest.iterate_trees()
    result = []
    for node in nodes:
        result.append(str(node))

    assert result == ['Root', 'A', 'B', 'Root', 'A', 'B']

def test_forest_with_multiple_root_trees():
    tree1 = Tree("A")
    tree2 = Tree("B")
    tree3 = Tree("C")

    forest = Forest([tree1, tree2, tree3])

    result = []
    for node in forest.iterate_trees():
        result.append(str(node))

    assert result == ['A', 'B', 'C']

def test_single_tree_with_nested_nodes():
    tree = Tree("Root")
    node_a = tree.add_child(tree.root, "A")
    node_b = tree.add_child(node_a, "B")
    node_c = tree.add_child(node_b, "C")
    node_d = tree.add_child(node_c, "D")

    forest = Forest([tree])

    result = []
    for node in forest.iterate_trees():
        result.append(str(node))

    assert result == ['Root', 'A', 'B', 'C', 'D']

