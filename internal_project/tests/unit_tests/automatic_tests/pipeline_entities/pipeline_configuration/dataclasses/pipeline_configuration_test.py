import unittest

from packaging.version import Version

from data_structures.tree.tree import Tree
from data_structures.tree.tree_node import TreeNode
from pipeline_entities.component_info.dataclasses.pipeline_component_info import PipelineComponentInfo
from pipeline_entities.components.dynamic_management.component_registry import ComponentRegistry
from pipeline_entities.pipeline_configuration.dataclasses.pipeline_configuration import PipelineConfiguration
from pipeline_entities.pipeline_configuration.dataclasses.pipeline_configuration_data import PipelineConfigurationData


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        ComponentRegistry.register_default_components()


    def test_all_attributes_correctly_set(self):
        data: PipelineConfigurationData = PipelineConfigurationData()

        data.name = "\"Test Data\""
        data.supported_program_version = "Version(\"1.0.3\")"
        data.components = "Tree(\"\"\"\ndummy\n dummy\n dummy\n\"\"\")"

        data.additional_values["test1"] = "Tree(None)"

        pipeline_configuration: PipelineConfiguration = PipelineConfiguration(data)

        self.assertEqual("Test Data", pipeline_configuration.name)
        self.assertEqual(Version("1.0.3"), pipeline_configuration.supported_program_version)

        dummy_info: PipelineComponentInfo = ComponentRegistry.get_component("dummy")
        root_node: TreeNode[PipelineComponentInfo] = TreeNode(dummy_info)
        child_node_1: TreeNode[PipelineComponentInfo] = TreeNode(dummy_info)
        child_node_2: TreeNode[PipelineComponentInfo] = TreeNode(dummy_info)
        root_node.add_child_nodes([child_node_1, child_node_2])

        self.assertEqual(Tree(root_node), pipeline_configuration.components)

        self.assertEqual({'test1': Tree(None)}, pipeline_configuration.additional_values)


    def test_only_required_attributes_set(self):
        data: PipelineConfigurationData = PipelineConfigurationData()

        data.supported_program_version = "Version(\"1.0.3\")"
        data.components = "Tree(\"\"\"\ndummy\n dummy\n dummy\n\"\"\")"

        pipeline_configuration: PipelineConfiguration = PipelineConfiguration(data)

        self.assertEqual(None, pipeline_configuration.name)
        self.assertEqual(Version("1.0.3"), pipeline_configuration.supported_program_version)

        dummy_info: PipelineComponentInfo = ComponentRegistry.get_component("dummy")
        root_node: TreeNode[PipelineComponentInfo] = TreeNode(dummy_info)
        child_node_1: TreeNode[PipelineComponentInfo] = TreeNode(dummy_info)
        child_node_2: TreeNode[PipelineComponentInfo] = TreeNode(dummy_info)
        root_node.add_child_nodes([child_node_1, child_node_2])

        self.assertEqual(Tree(root_node), pipeline_configuration.components)

        self.assertEqual({}, pipeline_configuration.additional_values)


    def test_supported_program_version_not_set(self):
        data: PipelineConfigurationData = PipelineConfigurationData()

        #data.supported_program_version = "Version(\"1.0.3\")"
        data.components = "Tree(\"\"\"\ndummy\n dummy\n dummy\n\"\"\")"

        with self.assertRaises(ValueError):
            PipelineConfiguration(data)


    def test_components_not_set(self):
        data: PipelineConfigurationData = PipelineConfigurationData()

        data.supported_program_version = "Version(\"1.0.3\")"
        #data.components = "Tree(\"\"\"\ndummy\n dummy\n dummy\n\"\"\")"

        with self.assertRaises(ValueError):
            PipelineConfiguration(data)


    def test_name_with_wrong_type(self):
        data: PipelineConfigurationData = PipelineConfigurationData()

        data.name = "Tree(None)"
        data.supported_program_version = "Version(\"1.0.3\")"
        data.components = "Tree(\"\"\"\ndummy\n dummy\n dummy\n\"\"\")"

        with self.assertRaises(TypeError):
            PipelineConfiguration(data)


    def test_supported_program_version_with_wrong_type(self):
        data: PipelineConfigurationData = PipelineConfigurationData()

        data.supported_program_version = "Tree(None)"
        data.components = "Tree(\"\"\"\ndummy\n dummy\n dummy\n\"\"\")"

        with self.assertRaises(TypeError):
            PipelineConfiguration(data)


    def test_components_with_wrong_type(self):
        data: PipelineConfigurationData = PipelineConfigurationData()

        data.supported_program_version = "Tree(None)"
        data.components = "False"

        with self.assertRaises(TypeError):
            PipelineConfiguration(data)


if __name__ == '__main__':
    unittest.main()
