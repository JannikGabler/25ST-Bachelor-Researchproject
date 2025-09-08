import textwrap
import unittest

from general_data_structures.directional_acyclic_graph.directional_acyclic_graph import DirectionalAcyclicGraph
from general_data_structures.directional_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode


class InitFromString(unittest.TestCase):
    def test_empty(self):
        input_string: str = ""

        result: DirectionalAcyclicGraph = DirectionalAcyclicGraph(input_string)
        expected: DirectionalAcyclicGraph = DirectionalAcyclicGraph()

        self.assertEqual(expected, result)

    
    def test_single_node_with_no_properties(self):
        input_string: str = textwrap.dedent("""
            0 = Type1
            """)

        result: DirectionalAcyclicGraph = DirectionalAcyclicGraph(input_string)

        expected: DirectionalAcyclicGraph = DirectionalAcyclicGraph()
        expected.add_node(DirectionalAcyclicGraphNode(("0", "Type1", {})))

        self.assertEqual(expected, result)
    

    def test_single_node_with_properties(self):
        input_string: str = textwrap.dedent("""
            masoas = Type1 a
                property1 = value1
                    + value2
                property2 = value3
            """)

        result: DirectionalAcyclicGraph = DirectionalAcyclicGraph(input_string)

        expected: DirectionalAcyclicGraph = DirectionalAcyclicGraph()
        expected.add_node(DirectionalAcyclicGraphNode(("masoas", "Type1 a", {'property1': 'value1\n+ value2', 'property2': 'value3'})))

        self.assertEqual(expected, result)


    def test_two_nodes_with_no_edges(self):
        input_string: str = textwrap.dedent("""
            node1 = Type1
                property1 = value1
                property2 = value2
            
            node2 = Type1
                property1 = value3
                property2 = value4
            """)

        result: DirectionalAcyclicGraph = DirectionalAcyclicGraph(input_string)

        expected: DirectionalAcyclicGraph = DirectionalAcyclicGraph()
        expected.add_node(DirectionalAcyclicGraphNode(("node1", "Type1", {'property1': 'value1', 'property2': 'value2'})))
        expected.add_node(DirectionalAcyclicGraphNode(("node2", "Type1", {'property1': 'value3', 'property2': 'value4'})))

        self.assertEqual(expected, result)


    def test_two_nodes_with_predecessors(self):
        input_string: str = textwrap.dedent("""
            node1 = Type1
                property1 = value1
                property2 = value2

            node2 = Type1
                property1 = value3
                predecessors = ["node1"]
                property2 = value4
            """)

        result: DirectionalAcyclicGraph = DirectionalAcyclicGraph(input_string)

        expected: DirectionalAcyclicGraph = DirectionalAcyclicGraph()
        node1: DirectionalAcyclicGraphNode = DirectionalAcyclicGraphNode(("node1", "Type1", {'property1': 'value1', 'property2': 'value2'}))
        node2: DirectionalAcyclicGraphNode = DirectionalAcyclicGraphNode(("node2", "Type1", {'property1': 'value3', 'property2': 'value4'}))
        expected.add_node(node1)
        expected.add_node(node2)
        expected.add_edge(node1, node2)

        self.assertEqual(expected, result)


    def test_two_nodes_with_successors(self):
        input_string: str = textwrap.dedent("""
            node1 = Type1
                property1 = value1
                successors = ["node2"]
                property2 = value2

            node2 = Type1
                property1 = value3
                property2 = value4
            """)

        result: DirectionalAcyclicGraph = DirectionalAcyclicGraph(input_string)

        expected: DirectionalAcyclicGraph = DirectionalAcyclicGraph()
        node1: DirectionalAcyclicGraphNode = DirectionalAcyclicGraphNode(("node1", "Type1", {'property1': 'value1', 'property2': 'value2'}))
        node2: DirectionalAcyclicGraphNode = DirectionalAcyclicGraphNode(("node2", "Type1", {'property1': 'value3', 'property2': 'value4'}))
        expected.add_node(node1)
        expected.add_node(node2)
        expected.add_edge(node1, node2)

        self.assertEqual(expected, result)


    def test_large_graph(self):
        input_string: str = textwrap.dedent("""
            node1 = Type6
                property1 = value1

            node2.1 = Type7
                property2 = value2
                predecessors = ["node1"]
            
            node2.2 = Type9
                predecessors = ["node1"]
                successors = ["node3.1"]
                
            node3.1 = Type41691
                predecessors = ["node2.1"]
            
            node3.2 = Type26
                property2 = value2
                predecessors = ["node2.2"]
                
            node4 = Type126
                predecessors = ["node3.1", "node3.2"]
                property3 = value3
            """)

        result: DirectionalAcyclicGraph = DirectionalAcyclicGraph(input_string)

        expected: DirectionalAcyclicGraph = DirectionalAcyclicGraph()

        node1: DirectionalAcyclicGraphNode = DirectionalAcyclicGraphNode(("node1", "Type6", {'property1': 'value1'}))
        node2_1: DirectionalAcyclicGraphNode = DirectionalAcyclicGraphNode(("node2.1", "Type7", {'property2': 'value2'}))
        node2_2: DirectionalAcyclicGraphNode = DirectionalAcyclicGraphNode(("node2.2", "Type9", {}))
        node3_1: DirectionalAcyclicGraphNode = DirectionalAcyclicGraphNode(("node3.1", "Type41691", {}))
        node3_2: DirectionalAcyclicGraphNode = DirectionalAcyclicGraphNode(("node3.2", "Type26", {'property2': 'value2'}))
        node4: DirectionalAcyclicGraphNode = DirectionalAcyclicGraphNode(("node4", "Type126", {'property3': 'value3'}))

        expected.add_node(node1)
        expected.add_node(node2_1)
        expected.add_node(node2_2)
        expected.add_node(node3_1)
        expected.add_node(node3_2)
        expected.add_node(node4)

        expected.add_edge(node1, node2_1)
        expected.add_edge(node1, node2_2)
        expected.add_edge(node2_1, node3_1)
        expected.add_edge(node2_2, node3_1)
        expected.add_edge(node2_2, node3_2)
        expected.add_edge(node3_1, node4)
        expected.add_edge(node3_2, node4)

        self.assertEqual(expected, result)



if __name__ == '__main__':
    unittest.main()
