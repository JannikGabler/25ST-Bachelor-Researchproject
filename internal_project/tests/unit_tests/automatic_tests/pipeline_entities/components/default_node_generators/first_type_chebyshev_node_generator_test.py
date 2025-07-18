import unittest
import jax.numpy as jnp

from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component import PipelineComponent
from pipeline_entities.pipeline.component_entities.default_components.default_node_generators.first_type_chebyshev_node_generator import \
    FirstTypeChebyshevNodeGenerator
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from utils.jax_utils import JaxUtils


class MyTestCase(unittest.TestCase):
    def test_normal_case_1(self):
        pipeline_data: PipelineData = PipelineData()
        pipeline_data.data_type = jnp.float16
        pipeline_data.node_count = 5
        pipeline_data.interpolation_interval = jnp.array([-1, 1], dtype=jnp.float16)

        expected_nodes = jnp.array([jnp.float16("0.9510565163"), jnp.float16("0.5877852523"), jnp.float16("0"),
                                    jnp.float16("-0.5877852523"), jnp.float16("-0.9510565163")], dtype=jnp.float16)

        generator: PipelineComponent = FirstTypeChebyshevNodeGenerator(pipeline_data)
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

        expected_nodes = jnp.array([0.9238795325, 0.3826834324, -0.3826834324, -0.9238795325], dtype=jnp.float32)

        generator: PipelineComponent = FirstTypeChebyshevNodeGenerator(pipeline_data)
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
        pipeline_data.interpolation_interval = jnp.array([-6, 1], dtype=jnp.float16)

        expected_nodes: jnp.ndarray = jnp.array([jnp.float16("0.9122476926"), jnp.float16("0.2364101886"),
                                        jnp.float16("-0.9814069131"), jnp.float16("-2.5"),
                                        jnp.float16("-4.018593087"), jnp.float16("-5.236410189"),
                                        jnp.float16("-5.912247693")], dtype=jnp.float16)

        generator: PipelineComponent = FirstTypeChebyshevNodeGenerator(pipeline_data)
        generator.perform_action()

        self.assertEqual(jnp.float16, pipeline_data.data_type)
        self.assertEqual(7, pipeline_data.node_count)
        self.assertTrue(jnp.array_equal(jnp.array([-6, 1]), pipeline_data.interpolation_interval))
        self.assertIsNone(pipeline_data.function_values)

        self.assertTrue(JaxUtils.all_close_enough(expected_nodes, pipeline_data.nodes, 1E-01))
        self.assertEqual(jnp.float16, pipeline_data.nodes.dtype)

        self.assertEqual({}, pipeline_data.additional_values)


    # def test_float16_generation_default_interval(self):
    #     interval: Tuple[float, float] = (-1, 1)
    #     node_count = 5
    #     data_type = jnp.float16
    #     generator: NodeGenerator = FirstTypeChebyshevNodeGenerator(interval, node_count, data_type)
    #
    #     expected = jnp.array([jnp.float16("0.9510565163"), jnp.float16("0.5877852523"), jnp.float16("0"), jnp.float16("-0.5877852523"), jnp.float16("-0.9510565163")], dtype=jnp.float16)
    #     result = generator.generate_nodes()
    #
    #     # Absolut tolerance is required because 0 appears as a value
    #     self.assertTrue(jnp.allclose(expected, result, atol=1E-3))



    # def test_float16_generation_transformed_interval(self):
    #     interval: Tuple[float, float] = (-6, 1)
    #     node_count: int = 7
    #     data_type: type = jnp.float16
    #     generator: NodeGenerator = FirstTypeChebyshevNodeGenerator(interval, node_count, data_type)
    #
    #     expected: jnp.ndarray = jnp.array([jnp.float16("0.9122476926"), jnp.float16("0.2364101886"),
    #                                     jnp.float16("-0.9814069131"), jnp.float16("-2.5"),
    #                                     jnp.float16("-4.018593087"), jnp.float16("-5.236410189"),
    #                                     jnp.float16("-5.912247693")], dtype=jnp.float16)
    #     result = generator.generate_nodes()
    #
    #     self.assertTrue(jnp.allclose(expected, result, atol=1E-3))


    # def test_float32_generation(self):
    #     interval: Tuple = (-1, 1)
    #     node_count = 4
    #     data_type = jnp.float32
    #     generator: EquidistantNodeGenerator = EquidistantNodeGenerator(interval, node_count, data_type)
    #
    #     expected_nodes = jnp.array([jnp.float32("-1"), jnp.float32(-1/3), jnp.float32(1/3), jnp.float32("1")], dtype=jnp.float32)
    #     result = generator.generate_nodes()
    #
    #     self.assertTrue(jnp.allclose(expected_nodes, result, rtol=1E-04))


if __name__ == '__main__':
    unittest.main()
