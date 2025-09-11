import textwrap
import unittest

from packaging.version import Version

from general_data_structures.directional_acyclic_graph.directional_acyclic_graph import DirectionalAcyclicGraph
from general_data_structures.directional_acyclic_graph.directional_acyclic_graph_node import DirectionalAcyclicGraphNode
from general_data_structures.tree.tree import Tree
from pipeline_entities.pipeline.component_entities.component_info.dataclasses.pipeline_component_info import PipelineComponentInfo
from pipeline_entities.pipeline.component_entities.component_registry.component_registry import ComponentRegistry
from pipeline_entities.pipeline_execution.dataclasses.pipeline_component_instantiation_info import \
    PipelineComponentInstantiationInfo
from data_classes.pipeline_configuration.pipeline_configuration import PipelineConfiguration
from data_classes.pipeline_configuration.pipeline_configuration_data import PipelineConfigurationData
from setup_manager.internal_logic_setup_manager import InternalLogicSetupManager


class InitPipelineConfiguration(unittest.TestCase):
    def setUp(self) -> None:
        InternalLogicSetupManager.setup()


    def test_all_attributes_correctly_set(self):
        data: PipelineConfigurationData = PipelineConfigurationData()

        data.name = "\"Test Data\""
        data.supported_program_version = "Version(\"1.0.3\")"
        data.components = textwrap.dedent("""
            DirectionalAcyclicGraph(\"\"\"
            0=dummy
                property1=value1
            1=dummy
                predecessors=["0"]
            2=dummy
                predecessors=["1"]
                property2=value2
            \"\"\")
            """)

        data.additional_values["test1"] = "Tree(None)"

        pipeline_configuration: PipelineConfiguration = PipelineConfiguration(data)

        self.assertEqual("Test Data", pipeline_configuration.name)
        self.assertEqual(Version("1.0.3"), pipeline_configuration.supported_program_version)

        dummy_info: PipelineComponentInfo = ComponentRegistry.get_component("dummy")

        node_0_value: PipelineComponentInstantiationInfo = PipelineComponentInstantiationInfo('0', dummy_info, {'property1': 'value1'})
        node_1_value: PipelineComponentInstantiationInfo = PipelineComponentInstantiationInfo('1', dummy_info, {})
        node_2_value: PipelineComponentInstantiationInfo = PipelineComponentInstantiationInfo('2', dummy_info, {'property2': 'value2'})

        node_0: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo] = DirectionalAcyclicGraphNode(node_0_value)
        node_1: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo] = DirectionalAcyclicGraphNode(node_1_value)
        node_2: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo] = DirectionalAcyclicGraphNode(node_2_value)

        expected_graph: DirectionalAcyclicGraph[PipelineComponentInstantiationInfo] = DirectionalAcyclicGraph()
        expected_graph.add_node(node_0)
        expected_graph.add_node(node_1)
        expected_graph.add_node(node_2)

        expected_graph.add_edge(node_0, node_1)
        expected_graph.add_edge(node_1, node_2)

        self.assertEqual(expected_graph, pipeline_configuration.components)

        self.assertEqual({'test1': Tree(None)}, pipeline_configuration.additional_values)


    def test_only_required_attributes_set(self):
        data: PipelineConfigurationData = PipelineConfigurationData()

        data.supported_program_version = "Version(\"1.0.3\")"
        data.components = textwrap.dedent("""
                    DirectionalAcyclicGraph(\"\"\"
                    0=dummy
                    1=dummy
                        predecessors=["0"]
                    2=dummy
                        predecessors=["1"]
                    \"\"\")
                    """)

        pipeline_configuration: PipelineConfiguration = PipelineConfiguration(data)

        self.assertEqual(None, pipeline_configuration.name)
        self.assertEqual(Version("1.0.3"), pipeline_configuration.supported_program_version)

        dummy_info: PipelineComponentInfo = ComponentRegistry.get_component("dummy")

        node_0_value: PipelineComponentInstantiationInfo = PipelineComponentInstantiationInfo('0', dummy_info, {})
        node_1_value: PipelineComponentInstantiationInfo = PipelineComponentInstantiationInfo('1', dummy_info, {})
        node_2_value: PipelineComponentInstantiationInfo = PipelineComponentInstantiationInfo('2', dummy_info, {})

        node_0: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo] = DirectionalAcyclicGraphNode(node_0_value)
        node_1: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo] = DirectionalAcyclicGraphNode(node_1_value)
        node_2: DirectionalAcyclicGraphNode[PipelineComponentInstantiationInfo] = DirectionalAcyclicGraphNode(node_2_value)

        expected_graph: DirectionalAcyclicGraph[PipelineComponentInstantiationInfo] = DirectionalAcyclicGraph()
        expected_graph.add_node(node_0)
        expected_graph.add_node(node_1)
        expected_graph.add_node(node_2)

        expected_graph.add_edge(node_0, node_1)
        expected_graph.add_edge(node_1, node_2)

        self.assertEqual(expected_graph, pipeline_configuration.components)

        self.assertEqual({}, pipeline_configuration.additional_values)


    def test_supported_program_version_not_set(self):
        data: PipelineConfigurationData = PipelineConfigurationData()

        #data.supported_program_version = "Version(\"1.0.3\")"
        data.components = textwrap.dedent("""
                    DirectionalAcyclicGraph(\"\"\"
                    0=dummy
                    1=dummy
                        predecessors=["0"]
                    2=dummy
                        predecessors=["1"]
                    \"\"\")
                    """)

        with self.assertRaises(ValueError):
            PipelineConfiguration(data)


    def test_components_not_set(self):
        data: PipelineConfigurationData = PipelineConfigurationData()

        data.supported_program_version = "Version(\"1.0.3\")"
        # data.components = textwrap.dedent("""
        #             DirectionalAcyclicGraph(\"\"\"
        #             0=dummy
        #             1=dummy
        #                 predecessors=["0"]
        #             2=dummy
        #                 predecessors=["1"]
        #             \"\"\")
        #             """)

        with self.assertRaises(ValueError):
            PipelineConfiguration(data)


    def test_name_with_wrong_type(self):
        data: PipelineConfigurationData = PipelineConfigurationData()

        data.name = "5"
        data.supported_program_version = "Version(\"1.0.3\")"
        data.components = textwrap.dedent("""
                    DirectionalAcyclicGraph(\"\"\"
                    0=dummy
                    1=dummy
                        predecessors=["0"]
                    2=dummy
                        predecessors=["1"]
                    \"\"\")
                    """)

        with self.assertRaises(TypeError):
            PipelineConfiguration(data)


    def test_supported_program_version_with_wrong_type(self):
        data: PipelineConfigurationData = PipelineConfigurationData()

        data.name = "\"Test Data\""
        data.supported_program_version = "DirectionalAcyclicGraph()"
        data.components = textwrap.dedent("""
                    DirectionalAcyclicGraph(\"\"\"
                    0=dummy
                    1=dummy
                        predecessors=["0"]
                    2=dummy
                        predecessors=["1"]
                    \"\"\")
                    """)

        with self.assertRaises(TypeError):
            PipelineConfiguration(data)


    def test_components_with_wrong_type(self):
        data: PipelineConfigurationData = PipelineConfigurationData()

        data.supported_program_version = "Version(\"1.0.3\")"
        data.components = "Tree(None)"

        with self.assertRaises(TypeError):
            PipelineConfiguration(data)


if __name__ == '__main__':
    unittest.main()
