import unittest
import jax.numpy as jnp

from pipeline_entities.pipeline.component_entities.default_components.default_node_generators.equidistant_node_generator import EquidistantNodeGenerator
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from utils.jax_utils import JaxUtils


class MyTestCase(unittest.TestCase):
    def test_normal_case_1(self):
        pipeline_data: PipelineData = PipelineData()
        pipeline_data.data_type = jnp.float16
        pipeline_data.node_count = 5
        pipeline_data.interpolation_interval = jnp.array([-1, 1])

        expected_nodes = jnp.array([jnp.float16("-1"), jnp.float16("-0.5"), jnp.float16("0"), jnp.float16("0.5"), jnp.float16("1")], dtype=jnp.float16)

        generator: EquidistantNodeGenerator = EquidistantNodeGenerator(pipeline_data)
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
        pipeline_data.node_count = 7
        pipeline_data.interpolation_interval = jnp.array([-2, 2])

        expected_nodes = jnp.array([jnp.float32(-2), jnp.float32(-2 + 4/6), jnp.float32(-2 + 8/6),
                                    jnp.float32(0), jnp.float32(-2 + 16/6), jnp.float32(-2 + 20/6),
                                    jnp.float32(2)], dtype=jnp.float32)

        generator: EquidistantNodeGenerator = EquidistantNodeGenerator(pipeline_data)
        generator.perform_action()

        self.assertEqual(jnp.float32, pipeline_data.data_type)
        self.assertEqual(7, pipeline_data.node_count)
        self.assertTrue(jnp.array_equal(jnp.array([-2, 2]), pipeline_data.interpolation_interval))
        self.assertIsNone(pipeline_data.function_values)

        self.assertTrue(JaxUtils.all_close_enough(expected_nodes, pipeline_data.nodes, 1E-01))
        self.assertEqual(jnp.float32, pipeline_data.nodes.dtype)

        self.assertEqual({}, pipeline_data.additional_values)


if __name__ == '__main__':
    unittest.main()
