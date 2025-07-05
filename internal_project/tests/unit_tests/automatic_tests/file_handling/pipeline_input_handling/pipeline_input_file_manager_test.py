import tempfile
import textwrap
import unittest
from pathlib import Path

from file_handling.pipeline_input_handling.pipeline_input_file_manager import PipelineInputFileManager
from pipeline_entities.pipeline_input.pipeline_input_data import PipelineInputData


class LoadFromFile(unittest.TestCase):
    def test_load_with_all_attributes_set(self):
        file_content: bytes = textwrap.dedent("""\
                                        name="TestPipeline"
                                        data_type=jnp.float32
                                        node_count=5
                                        interpolation_interval=jnp.array([-1, 1])
                                        single_function_expression="x**2 + 1"
                                        multiple_function_expressions=[((0,1), 'x'), ((1,2), 'x**2')]
                                        sympy_function_expression_simplification=True
                                        §secret_token="abc123"
                                        custom_param=[1, 2, 3]
                                        function_callable=lambda x: x**2 + 3
                                        function_values=jnp.array([0.0, 1.0, 4.0, 9.0])
                                        """).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir + "/temp_file.py")

            with open(temp_file, "wb") as f:
                f.write(file_content)

            pipeline_input_data: PipelineInputData = PipelineInputFileManager.load_from_file(temp_file)

            self.assertEqual("\"TestPipeline\"", pipeline_input_data.name)
            self.assertEqual("jnp.float32", pipeline_input_data.data_type)
            self.assertEqual("5", pipeline_input_data.node_count)
            self.assertEqual("jnp.array([-1, 1])", pipeline_input_data.interpolation_interval)
            self.assertEqual("\"x**2 + 1\"", pipeline_input_data.function_expression)
            self.assertEqual("[((0,1), 'x'), ((1,2), 'x**2')]", pipeline_input_data.piecewise_function_expression)
            self.assertEqual("True", pipeline_input_data.sympy_function_expression_simplification)
            self.assertEqual("lambda x: x**2 + 3", pipeline_input_data.function_callable)
            self.assertEqual("jnp.array([0.0, 1.0, 4.0, 9.0])", pipeline_input_data.interpolation_values)

            self.assertEqual({"secret_token": "\"abc123\""}, pipeline_input_data.additional_directly_injected_values)
            self.assertEqual({"custom_param": "[1, 2, 3]"}, pipeline_input_data.additional_values)


    def test_load_empty_file(self):
        file_content: bytes = textwrap.dedent("""\
                                        # This is a comment
                                        """).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir + "/temp_file.py")

            with open(temp_file, "wb") as f:
                f.write(file_content)

            pipeline_input_data: PipelineInputData = PipelineInputFileManager.load_from_file(temp_file)

            self.assertEqual(None, pipeline_input_data.name)
            self.assertEqual(None, pipeline_input_data.data_type)
            self.assertEqual(None, pipeline_input_data.node_count)
            self.assertEqual(None, pipeline_input_data.interpolation_interval)
            self.assertEqual(None, pipeline_input_data.function_expression)
            self.assertEqual(None, pipeline_input_data.piecewise_function_expression)
            self.assertEqual(None, pipeline_input_data.sympy_function_expression_simplification)
            self.assertEqual(None, pipeline_input_data.function_callable)
            self.assertEqual(None, pipeline_input_data.interpolation_values)

            self.assertEqual({}, pipeline_input_data.additional_directly_injected_values)
            self.assertEqual({}, pipeline_input_data.additional_values)


    def test_dictionary_names_as_keys(self):
        file_content: bytes = textwrap.dedent("""\
                                        additional_values=5
                                        additional_directly_injected_values=9
                                        """).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir + "/temp_file.py")

            with open(temp_file, "wb") as f:
                f.write(file_content)

            pipeline_input_data: PipelineInputData = PipelineInputFileManager.load_from_file(temp_file)

            self.assertEqual(None, pipeline_input_data.name)
            self.assertEqual(None, pipeline_input_data.data_type)
            self.assertEqual(None, pipeline_input_data.node_count)
            self.assertEqual(None, pipeline_input_data.interpolation_interval)
            self.assertEqual(None, pipeline_input_data.function_expression)
            self.assertEqual(None, pipeline_input_data.piecewise_function_expression)
            self.assertEqual(None, pipeline_input_data.sympy_function_expression_simplification)
            self.assertEqual(None, pipeline_input_data.function_callable)
            self.assertEqual(None, pipeline_input_data.interpolation_values)

            self.assertEqual({}, pipeline_input_data.additional_directly_injected_values)
            self.assertEqual({"additional_values": "5", "additional_directly_injected_values": "9"}, pipeline_input_data.additional_values)


if __name__ == '__main__':
    unittest.main()
