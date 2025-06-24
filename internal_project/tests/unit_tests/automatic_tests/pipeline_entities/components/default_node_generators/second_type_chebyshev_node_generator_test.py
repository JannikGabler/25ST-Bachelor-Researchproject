import unittest
import jax.numpy as jnp

from pipeline_entities.components.abstracts.pipeline_component import PipelineComponent
from pipeline_entities.components.default_components.default_node_generators.second_type_chebyshev_node_generator import \
    SecondTypeChebyshevNodeGenerator
from pipeline_entities.data_transfer.pipeline_data import PipelineData
from utils.jax_utils import JaxUtils


class MyTestCase(unittest.TestCase):
    def test_normal_case_1(self):
        pipeline_data: PipelineData = PipelineData()
        pipeline_data.data_type = jnp.float16
        pipeline_data.node_count = 5
        pipeline_data.interpolation_interval = jnp.array([-1, 1], dtype=jnp.float16)

        expected_nodes = jnp.array([1, 0.7071067812, 0, -0.7071067812, -1], dtype=jnp.float16)

        generator: PipelineComponent = SecondTypeChebyshevNodeGenerator(pipeline_data)
        generator.perform_action()

        self.assertEqual(jnp.float16, pipeline_data.data_type)
        self.assertEqual(5, pipeline_data.node_count)
        self.assertTrue(jnp.array_equal(jnp.array([-1, 1]), pipeline_data.interpolation_interval))
        self.assertIsNone(pipeline_data.function_values)

        self.assertTrue(JaxUtils.all_close_enough(expected_nodes, pipeline_data.nodes, 1E-01))
        self.assertEqual(jnp.float16, pipeline_data.nodes.dtype)

        self.assertEqual({}, pipeline_data.additional_values)



    def test_normal_case_2(self):
        pipeline_data: PipelineData = PipelineData()
        pipeline_data.data_type = jnp.float32
        pipeline_data.node_count = 4
        pipeline_data.interpolation_interval = jnp.array([-1, 1], dtype=jnp.float32)

        expected_nodes = jnp.array([1, 0.5, -0.5, -1], dtype=jnp.float32)

        generator: PipelineComponent = SecondTypeChebyshevNodeGenerator(pipeline_data)
        generator.perform_action()

        self.assertEqual(jnp.float32, pipeline_data.data_type)
        self.assertEqual(4, pipeline_data.node_count)
        self.assertTrue(jnp.array_equal(jnp.array([-1, 1]), pipeline_data.interpolation_interval))
        self.assertIsNone(pipeline_data.function_values)

        self.assertTrue(JaxUtils.all_close_enough(expected_nodes, pipeline_data.nodes, 1E-01))
        self.assertEqual(jnp.float32, pipeline_data.nodes.dtype)

        self.assertEqual({}, pipeline_data.additional_values)



    def test_transformed_interval(self):
        pipeline_data: PipelineData = PipelineData()
        pipeline_data.data_type = jnp.float16
        pipeline_data.node_count = 7
        pipeline_data.interpolation_interval = jnp.array([-4, 1], dtype=jnp.float16)

        expected_nodes: jnp.ndarray = jnp.array([1, 0.6650635095, -0.25, -1.5, -2.75, -3.665063509, -4], dtype=jnp.float16)

        generator: PipelineComponent = SecondTypeChebyshevNodeGenerator(pipeline_data)
        generator.perform_action()

        self.assertEqual(jnp.float16, pipeline_data.data_type)
        self.assertEqual(7, pipeline_data.node_count)
        self.assertTrue(jnp.array_equal(jnp.array([-4, 1]), pipeline_data.interpolation_interval))
        self.assertIsNone(pipeline_data.function_values)

        self.assertTrue(JaxUtils.all_close_enough(expected_nodes, pipeline_data.nodes, 1E-01))
        self.assertEqual(jnp.float16, pipeline_data.nodes.dtype)

        self.assertEqual({}, pipeline_data.additional_values)



if __name__ == '__main__':
    unittest.main()
