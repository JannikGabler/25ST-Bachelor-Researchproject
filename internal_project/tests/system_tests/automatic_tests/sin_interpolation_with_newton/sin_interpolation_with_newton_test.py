import unittest
import textwrap
import tempfile
from fractions import Fraction

import jax.numpy as jnp

from pathlib import Path

from file_handling.pipeline_input_handling.pipeline_input_file_manager import PipelineInputFileManager
from pipeline_entities.pipeline.component_entities.component_registry.component_registry import ComponentRegistry
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline.pipeline import Pipeline
from pipeline_entities.pipeline_execution.output.pipeline_component_execution_report import PipelineComponentExecutionReport
from pipeline_entities.pipeline_execution.dataclasses.pipeline_component_instantiation_info import \
    PipelineComponentInstantiationInfo
from pipeline_entities.large_data_classes.pipeline_configuration.pipeline_configuration_data import \
    PipelineConfigurationData
from file_handling.pipeline_configuration_handling.pipeline_configuration_file_manager import \
    PipelineConfigurationFileManager
from pipeline_entities.large_data_classes.pipeline_configuration.pipeline_configuration import PipelineConfiguration
from pipeline_entities.large_data_classes.pipeline_input.pipeline_input import PipelineInput
from pipeline_entities.large_data_classes.pipeline_input.pipeline_input_data import PipelineInputData
from pipeline_entities.pipeline.pipeline_builder.pipeline_builder import PipelineBuilder
from pipeline_entities.pipeline_execution.pipeline_manager.pipeline_manager import PipelineManager
from setup_manager.internal_logic_setup_manager import InternalLogicSetupManager
from system_tests.test_utils.evaluation_test_utils import EvaluationTestUtils
from system_tests.test_utils.interpolation_test_utils import InterpolationTestUtils
from system_tests.test_utils.nodes_test_utils import NodesTestUtils
from utils.jax_utils import JaxUtils


class ChebyshevPoints2Case(unittest.TestCase):
    _pipeline_manager_: PipelineManager


    def setUp(self):
        InternalLogicSetupManager.setup()

        pipeline_configuration_file_content: bytes = textwrap.dedent("""\
            name="DemoPipeline"
            supported_program_version=Version(\"1.0.0\")
            components=DirectionalAcyclicGraph(\"\"\"
                0=Base Input
                    test_attribute=jax.numpy.float32
                1=Function Expression Input
                    predecessors=["0"]
                2=Chebyshev2 Node Generator
                    predecessors=["1"]
                3=Interpolation Values Evaluator
                    predecessors=["2"]
                4=Newton Interpolation
                    predecessors=["3"]
                \"\"\")
            extra_value=True
            """).encode("utf-8")

        temp_dir = tempfile.TemporaryDirectory()
        temp_pipeline_configuration_file = Path(temp_dir.name + "/pipeline_configuration.ini")

        with open(temp_pipeline_configuration_file, "wb") as f:
            f.write(pipeline_configuration_file_content)

        pipeline_configuration_data: PipelineConfigurationData = PipelineConfigurationFileManager.load_from_file(temp_pipeline_configuration_file)
        pipeline_configuration: PipelineConfiguration = PipelineConfiguration(pipeline_configuration_data)


        pipeline_input_file_content: bytes = textwrap.dedent("""\
            name="TestPipeline"
            data_type=jax.numpy.float32
            node_count=37
            interpolation_interval=jax.numpy.array([-1, 1])
            function_expression="sin(10 * x)"
            sympy_function_expression_simplification=True
            """).encode("utf-8")

        temp_pipeline_input_file = Path(temp_dir.name + "/pipeline_input.ini")

        with open(temp_pipeline_input_file, "wb") as f:
            f.write(pipeline_input_file_content)

        pipeline_input_data: PipelineInputData = PipelineInputFileManager.load_from_file(temp_pipeline_input_file)
        pipeline_input: PipelineInput = PipelineInput(pipeline_input_data)


        pipeline: Pipeline = PipelineBuilder.build(pipeline_configuration, pipeline_input)

        self._pipeline_manager_ = PipelineManager(pipeline)

        self._pipeline_manager_.execute_all()


    def test_base_input_output(self):
        report: PipelineComponentExecutionReport = self._pipeline_manager_.get_component_execution_report("0")


        instantiation_info: PipelineComponentInstantiationInfo = report.component_instantiation_info

        self.assertEqual("0", instantiation_info.component_name)
        self.assertEqual(ComponentRegistry.get_component("Base Input"), instantiation_info.component)
        #self.assertEqual({'test_attribute': jnp.float32}, instantiation_info.component_specific_arguments)


        output: PipelineData = report.component_output

        self.assertEqual(jnp.float32, output.data_type)
        self.assertEqual(37, output.node_count)
        self.assertTrue(jnp.array_equal(jnp.array([-1, 1]), output.interpolation_interval))
        self.assertIsNone(output.original_function)
        self.assertIsNone(output.interpolation_nodes)
        self.assertIsNone(output.interpolation_values)
        self.assertIsNone(output.interpolant)
        self.assertEqual({}, output.additional_values)


        self.assertTrue(0 < report.component_init_time <= 0.5)
        self.assertTrue(0 < report.component_execution_time <= 0.5)


    def test_function_expression_input_output(self):
        report: PipelineComponentExecutionReport = self._pipeline_manager_.get_component_execution_report("1")


        instantiation_info: PipelineComponentInstantiationInfo = report.component_instantiation_info

        self.assertEqual("1", instantiation_info.component_name)
        self.assertEqual(ComponentRegistry.get_component("Function Expression Input"), instantiation_info.component)
        # self.assertEqual({'test_attribute': jnp.float32}, instantiation_info.component_specific_arguments)


        output: PipelineData = report.component_output

        self.assertEqual(jnp.float32, output.data_type)
        self.assertEqual(37, output.node_count)
        self.assertTrue(jnp.array_equal(jnp.array([-1, 1]), output.interpolation_interval))
        self.assertIsNotNone(output.original_function)

        x_values = jnp.linspace(-1, 1, 1000)
        self.assertTrue(JaxUtils.all_close_enough(jnp.sin(10 * x_values), output.original_function(x_values), rtol=1E-04))

        self.assertIsNone(output.interpolation_nodes)
        self.assertIsNone(output.interpolation_values)
        self.assertIsNone(output.interpolant)
        self.assertEqual({}, output.additional_values)


        self.assertTrue(0 < report.component_init_time <= 0.5)
        self.assertTrue(0 < report.component_execution_time <= 0.5)


    def test_chebyshev_2_node_generator_output(self):
        report: PipelineComponentExecutionReport = self._pipeline_manager_.get_component_execution_report("2")


        instantiation_info: PipelineComponentInstantiationInfo = report.component_instantiation_info

        self.assertEqual("2", instantiation_info.component_name)
        self.assertEqual(ComponentRegistry.get_component("Chebyshev2 Node Generator"), instantiation_info.component)
        # self.assertEqual({'test_attribute': jnp.float32}, instantiation_info.component_specific_arguments)


        output: PipelineData = report.component_output

        self.assertEqual(jnp.float32, output.data_type)
        self.assertEqual(37, output.node_count)
        self.assertTrue(jnp.array_equal(jnp.array([-1, 1]), output.interpolation_interval))
        self.assertIsNotNone(output.original_function)

        x_values = jnp.linspace(-1, 1, 1000)
        self.assertIsNotNone(output.original_function)
        self.assertTrue(JaxUtils.all_close_enough(jnp.sin(10 * x_values), output.original_function(x_values), rtol=1E-04))

        expected_nodes = jnp.cos(jnp.arange(0, 37, dtype=output.data_type) * (jnp.pi / 36))

        self.assertIsNotNone(output.interpolation_nodes)
        self.assertTrue(JaxUtils.all_close_enough(expected_nodes, output.interpolation_nodes, atol=1E-07, rtol=1E-04))

        self.assertIsNone(output.interpolation_values)
        self.assertIsNone(output.interpolant)
        self.assertEqual({}, output.additional_values)


        self.assertTrue(0 < report.component_init_time <= 0.5)
        self.assertTrue(0 < report.component_execution_time <= 0.5)


    def test_interpolation_values_evaluator_output(self):
        report: PipelineComponentExecutionReport = self._pipeline_manager_.get_component_execution_report("3")


        instantiation_info: PipelineComponentInstantiationInfo = report.component_instantiation_info

        self.assertEqual("3", instantiation_info.component_name)
        self.assertEqual(ComponentRegistry.get_component("Interpolation Values Evaluator"), instantiation_info.component)
        # self.assertEqual({'test_attribute': jnp.float32}, instantiation_info.component_specific_arguments)


        output: PipelineData = report.component_output

        self.assertEqual(jnp.float32, output.data_type)
        self.assertEqual(37, output.node_count)
        self.assertTrue(jnp.array_equal(jnp.array([-1, 1]), output.interpolation_interval))
        self.assertIsNotNone(output.original_function)

        x_values = jnp.linspace(-1, 1, 1000)
        self.assertIsNotNone(output.original_function)
        self.assertTrue(JaxUtils.all_close_enough(jnp.sin(10 * x_values), output.original_function(x_values), rtol=1E-04))

        expected_nodes = jnp.cos(jnp.arange(0, 37, dtype=output.data_type) * (jnp.pi / 36))

        self.assertIsNotNone(output.interpolation_nodes)
        self.assertTrue(JaxUtils.all_close_enough(expected_nodes, output.interpolation_nodes, atol=1E-07, rtol=1E-04))

        expected_interpolation_values: jnp.ndarray = jnp.sin(10 * expected_nodes)

        self.assertIsNotNone(output.interpolation_values)
        self.assertTrue(JaxUtils.all_close_enough(expected_interpolation_values, output.interpolation_values, atol=1E-07, rtol=1E-04))

        self.assertIsNone(output.interpolant)
        self.assertEqual({}, output.additional_values)


        self.assertTrue(0 < report.component_init_time <= 0.5)
        self.assertTrue(0 < report.component_execution_time <= 0.5)


    def test_newton_interpolation_output(self):
        report: PipelineComponentExecutionReport = self._pipeline_manager_.get_component_execution_report("4")


        instantiation_info: PipelineComponentInstantiationInfo = report.component_instantiation_info

        self.assertEqual("4", instantiation_info.component_name)
        self.assertEqual(ComponentRegistry.get_component("Newton Interpolation"), instantiation_info.component)
        # self.assertEqual({'test_attribute': jnp.float32}, instantiation_info.component_specific_arguments)


        output: PipelineData = report.component_output

        self.assertEqual(jnp.float32, output.data_type)
        self.assertEqual(37, output.node_count)
        self.assertTrue(jnp.array_equal(jnp.array([-1, 1]), output.interpolation_interval))
        self.assertIsNotNone(output.original_function)

        x_values = jnp.linspace(-1, 1, 1000)
        self.assertIsNotNone(output.original_function)
        self.assertTrue(JaxUtils.all_close_enough(jnp.sin(10 * x_values), output.original_function(x_values), rtol=1E-04))

        expected_fraction_nodes = NodesTestUtils.chebyshev_2_nodes((Fraction(-1), Fraction(1)), output.node_count)
        expected_nodes = jnp.array([float(frac) for frac in expected_fraction_nodes], dtype=output.data_type)

        self.assertIsNotNone(output.interpolation_nodes)
        self.assertTrue(JaxUtils.all_close_enough(expected_nodes, output.interpolation_nodes, atol=1E-07, rtol=1E-04))

        expected_interpolation_values: jnp.ndarray = jnp.sin(10 * expected_nodes)

        self.assertIsNotNone(output.interpolation_values)
        self.assertTrue(JaxUtils.all_close_enough(expected_interpolation_values, output.interpolation_values, atol=1E-06, rtol=1E-04))

        expected_divided_differences: list[Fraction] = InterpolationTestUtils.calc_divided_differences(expected_fraction_nodes, expected_interpolation_values)
        evaluation_fraction_array: list[Fraction] = [Fraction(i, 10) for i in range(-10, 11)]
        evaluation_float_array: jnp.ndarray = jnp.array([float(frac) for frac in evaluation_fraction_array], dtype=output.data_type)
        expected_fraction_values = EvaluationTestUtils.evaluate_newton_polynom(expected_divided_differences, expected_fraction_nodes, evaluation_fraction_array)
        expected_values = jnp.array([float(frac) for frac in expected_fraction_values], dtype=output.data_type)

        self.assertIsNotNone(output.interpolant)
        self.assertTrue(JaxUtils.all_close_enough(expected_values, output.interpolant.evaluate(evaluation_float_array), atol=1E-06, rtol=1E-04))

        self.assertEqual({}, output.additional_values)


        self.assertTrue(0 < report.component_init_time <= 0.5)
        self.assertTrue(0 < report.component_execution_time <= 0.5)



if __name__ == '__main__':
    unittest.main()
