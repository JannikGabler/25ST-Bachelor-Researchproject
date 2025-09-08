import math
import tempfile
import textwrap
import unittest
from pathlib import Path

import jax.numpy as jnp

from general_data_structures.tree.tree import Tree
from file_handling.dynamic_module_loading.dynamic_module_loader import DynamicModuleLoader
from pipeline_entities.large_data_classes.pipeline_input.pipeline_input import PipelineInput
from pipeline_entities.large_data_classes.pipeline_input.pipeline_input_data import PipelineInputData


class MyTestCase(unittest.TestCase):
    def tearDown(self) -> None:
        DynamicModuleLoader.unload_all_modules()



    def test_all_attributes_correctly_set(self):
        file_content: bytes = textwrap.dedent("""\
                                        import jax.numpy as jnp

                                        def test_function(x: jnp.ndarray) -> jnp.ndarray:
                                            return jnp.sin(x) * jnp.cos(x)
                                        """).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir + "/temp_file.py")

            with open(temp_file, "wb") as f:
                f.write(file_content)

            DynamicModuleLoader.load_directory(Path(temp_dir))

            data: PipelineInputData = PipelineInputData()
            data.name = "\"Test Data\""
            data.data_type = "jax.numpy.float32"
            data.node_count = "5"
            data.interpolation_interval = "jax.numpy.array([-1, 1], dtype=jax.numpy.float32)"
            data.function_expression = "\"sin(x) * cos(x)\""
            data.piecewise_function_expression = "[((-1, 0), \"sin(x)\"), ((0, 1), \"cos(x)\")]"
            data.sympy_function_expression_simplification = "True"
            data.function_callable = f"{temp_file.stem}.test_function"
            data.interpolation_values = "jax.numpy.array([1, 2, 3, 4, 5], dtype=jax.numpy.float32)"
            data.additional_values["test1"] = "Tree(None)"
            data.additional_directly_injected_values["test2"] = "math.pi"

            pipeline_input: PipelineInput = PipelineInput(data)

            self.assertEqual("Test Data", pipeline_input.name)
            self.assertEqual(jnp.float32, pipeline_input.data_type)
            self.assertEqual(5, pipeline_input.node_count)
            self.assertTrue(jnp.array_equal(jnp.array([-1, 1], dtype=jnp.float32), pipeline_input.interpolation_interval))
            self.assertEqual("sin(x) * cos(x)", pipeline_input.function_expression)
            self.assertEqual([((-1, 0), "sin(x)"), ((0, 1), "cos(x)")], pipeline_input.piecewise_function_expression)
            self.assertEqual(True, pipeline_input.sympy_function_expression_simplification)

            callable_input: jnp.ndarray = jnp.linspace(-50000, 50000, 1000)
            expected_callable_results: jnp.ndarray = jnp.sin(callable_input) * jnp.cos(callable_input)
            callable_results: jnp.ndarray = pipeline_input.function_callable(callable_input)
            self.assertTrue(jnp.array_equal(expected_callable_results, callable_results))

            self.assertTrue(jnp.array_equal(jnp.array([1, 2, 3, 4, 5], dtype=jnp.float32), pipeline_input.interpolation_values))
            self.assertEqual({'test1': Tree(None)}, pipeline_input.additional_values)
            self.assertEqual({'test2': math.pi}, pipeline_input.additional_directly_injected_values)


    def test_only_required_attributes_set(self):
        data: PipelineInputData = PipelineInputData()

        data.data_type = "jax.numpy.float32"
        data.node_count = "5"
        data.interpolation_interval = "jax.numpy.array([-1, 1], dtype=jax.numpy.float32)"

        pipeline_input: PipelineInput = PipelineInput(data)

        self.assertEqual(None, pipeline_input.name)  # Changed
        self.assertEqual(jnp.float32, pipeline_input.data_type)
        self.assertEqual(5, pipeline_input.node_count)
        self.assertTrue(jnp.array_equal(jnp.array([-1, 1], dtype=jnp.float32), pipeline_input.interpolation_interval))
        self.assertEqual(None, pipeline_input.function_expression)
        self.assertEqual(None, pipeline_input.piecewise_function_expression)
        self.assertEqual(None, pipeline_input.sympy_function_expression_simplification)
        self.assertEqual(None, pipeline_input.function_callable)
        self.assertEqual(None, pipeline_input.interpolation_values)

        self.assertEqual({}, pipeline_input.additional_values)
        self.assertEqual({}, pipeline_input.additional_directly_injected_values)


    def test_data_type_not_set(self):
        data: PipelineInputData = PipelineInputData()

        # data.data_type = "jax.numpy.float32"
        data.node_count = "5"
        data.interpolation_interval = "jax.numpy.array([-1, 1], dtype=jax.numpy.float32)"

        with self.assertRaises(ValueError):
            PipelineInput(data)


    def test_node_count_not_set(self):
        data: PipelineInputData = PipelineInputData()

        data.data_type = "jax.numpy.float32"
        # data.node_count = "5"
        data.interpolation_interval = "jax.numpy.array([-1, 1], dtype=jax.numpy.float32)"

        with self.assertRaises(ValueError):
            PipelineInput(data)


    def test_interpolation_interval_not_set(self):
        data: PipelineInputData = PipelineInputData()

        data.data_type = "jax.numpy.float32"
        data.node_count = "5"
        # data.interpolation_interval = "jax.numpy.array([-1, 1], dtype=jax.numpy.float32)"

        with self.assertRaises(ValueError):
            PipelineInput(data)


    def test_name_with_wrong_type(self):
        data: PipelineInputData = PipelineInputData()

        data.name = "6" # wrong type
        data.data_type = "jax.numpy.float32"
        data.node_count = "5"
        data.interpolation_interval = "jax.numpy.array([-1, 1], dtype=jax.numpy.float32)"

        with self.assertRaises(TypeError):
            PipelineInput(data)


    def test_data_type_with_wrong_type(self):
        data: PipelineInputData = PipelineInputData()

        data.data_type = "\"I am not a type :D\"" # wrong type
        data.node_count = "5"
        data.interpolation_interval = "jax.numpy.array([-1, 1], dtype=jax.numpy.float32)"

        with self.assertRaises(TypeError):
            PipelineInput(data)


    def test_node_count_with_wrong_type(self):
        data: PipelineInputData = PipelineInputData()

        data.data_type = "jax.numpy.float32"
        data.node_count = "Tree(None)" # wrong type
        data.interpolation_interval = "jax.numpy.array([-1, 1], dtype=jax.numpy.float32)"

        with self.assertRaises(TypeError):
            PipelineInput(data)


    def test_interpolation_interval_with_wrong_type(self):
        data: PipelineInputData = PipelineInputData()

        data.data_type = "jax.numpy.float32"
        data.node_count = "5"
        data.interpolation_interval = "False"

        with self.assertRaises(TypeError):
            PipelineInput(data)


    def test_single_function_expression_with_wrong_type(self):
        data: PipelineInputData = PipelineInputData()

        data.data_type = "jax.numpy.float32"
        data.node_count = "5"
        data.interpolation_interval = "jax.numpy.array([-1, 1], dtype=jax.numpy.float32)"
        data.function_expression = "[]" # wrong type

        with self.assertRaises(TypeError):
            PipelineInput(data)


    def test_multiple_function_expressions_with_wrong_type(self):
        data: PipelineInputData = PipelineInputData()

        data.data_type = "jax.numpy.float32"
        data.node_count = "5"
        data.interpolation_interval = "jax.numpy.array([-1, 1], dtype=jax.numpy.float32)"
        data.piecewise_function_expression = "[(5, \"cos(x)\")]" # wrong type

        with self.assertRaises(TypeError):
            PipelineInput(data)

    def test_sympy_function_expression_simplification_with_wrong_type(self):
        data: PipelineInputData = PipelineInputData()

        data.data_type = "jax.numpy.float32"
        data.node_count = "5"
        data.interpolation_interval = "jax.numpy.array([-1, 1], dtype=jax.numpy.float32)"
        data.sympy_function_expression_simplification = "\"True\"" # wrong type

        with self.assertRaises(TypeError):
            PipelineInput(data)


    def test_function_callable_with_wrong_type(self):
        data: PipelineInputData = PipelineInputData()

        data.data_type = "jax.numpy.float32"
        data.node_count = "5"
        data.interpolation_interval = "jax.numpy.array([-1, 1], dtype=jax.numpy.float32)"
        data.function_callable = "{\"key\": 5}" # wrong type

        with self.assertRaises(TypeError):
            PipelineInput(data)


    def test_function_values_with_wrong_type(self):
        data: PipelineInputData = PipelineInputData()

        data.data_type = "jax.numpy.float32"
        data.node_count = "5"
        data.interpolation_interval = "jax.numpy.array([-1, 1], dtype=jax.numpy.float32)"
        data.interpolation_values = "[6, 8, 2]" # wrong type

        with self.assertRaises(TypeError):
            PipelineInput(data)


if __name__ == '__main__':
    unittest.main()
