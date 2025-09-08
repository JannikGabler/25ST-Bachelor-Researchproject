import tempfile
import textwrap
import unittest
from pathlib import Path

from file_handling.pipeline_configuration_handling.pipeline_configuration_file_manager import \
    PipelineConfigurationFileManager
from pipeline_entities.large_data_classes.pipeline_configuration.pipeline_configuration_data import PipelineConfigurationData


class MyTestCase(unittest.TestCase):
    def test_load_with_all_attributes_set(self):
        file_content: bytes = textwrap.dedent("""
                                        name="TestPipeline"
                                        supported_program_version=Version(\"1.0.0\")
                                        components=DirectionalAcyclicGraph(\"\"\"
                                            root=Root
                                            
                                            child=Child
                                                predecessors=["root"]
                                            \"\"\")
                                        extra_value=True
                                        """).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir + "/temp_file.py")

            with open(temp_file, "wb") as f:
                f.write(file_content)

            pipeline_configuration_data: PipelineConfigurationData = PipelineConfigurationFileManager.load_from_file(temp_file)

            # print(repr("Tree(\"\"\"\nroot\n    child\n\"\"\")"))
            # print(repr(pipeline_configuration_data.components))

            self.assertEqual("\"TestPipeline\"", pipeline_configuration_data.name)
            self.assertEqual("Version(\"1.0.0\")", pipeline_configuration_data.supported_program_version)
            self.assertEqual("DirectionalAcyclicGraph(\"\"\"\nroot=Root\n\nchild=Child\n    predecessors=[\"root\"]\n\"\"\")", pipeline_configuration_data.components)

            self.assertEqual({"extra_value": "True"}, pipeline_configuration_data.additional_values)


    def test_load_empty_file(self):
        file_content: bytes = textwrap.dedent("""\
                                        # This is a comment
                                        """).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir + "/temp_file.py")

            with open(temp_file, "wb") as f:
                f.write(file_content)

            pipeline_configuration_data: PipelineConfigurationData = PipelineConfigurationFileManager.load_from_file(temp_file)

            self.assertEqual(None, pipeline_configuration_data.name)
            self.assertEqual(None, pipeline_configuration_data.supported_program_version)
            self.assertEqual(None, pipeline_configuration_data.components)

            self.assertEqual({}, pipeline_configuration_data.additional_values)


    def test_dictionary_names_as_keys(self):
        file_content: bytes = textwrap.dedent("""\
                                        additional_values=5
                                        """).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir + "/temp_file.py")

            with open(temp_file, "wb") as f:
                f.write(file_content)

            pipeline_configuration_data: PipelineConfigurationData = PipelineConfigurationFileManager.load_from_file(temp_file)

            self.assertEqual(None, pipeline_configuration_data.name)
            self.assertEqual(None, pipeline_configuration_data.supported_program_version)
            self.assertEqual(None, pipeline_configuration_data.components)

            self.assertEqual({"additional_values": "5"}, pipeline_configuration_data.additional_values)


if __name__ == '__main__':
    unittest.main()
