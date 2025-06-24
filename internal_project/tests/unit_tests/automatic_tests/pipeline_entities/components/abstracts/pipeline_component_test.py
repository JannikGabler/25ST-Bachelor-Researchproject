import unittest

from pipeline.components.abstracts.pipeline_component import PipelineComponent
from pipeline.components.default_components.default_node_generators.equidistant_node_generator import EquidistantNodeGenerator
from pipeline.components.default_components.default_node_generators.first_type_chebyshev_node_generator import \
    FirstTypeChebyshevNodeGenerator
from pipeline.components.default_components.default_node_generators.second_type_chebyshev_node_generator import \
    SecondTypeChebyshevNodeGenerator
from pipeline.components.enums.component_type import ComponentType


class MyTestCase(unittest.TestCase):
    def test_attributes_of_equidistant_node_generator(self):
        generator: PipelineComponent = EquidistantNodeGenerator((-1, 1), 5, float)

        expected_id: str = "Equidistant"
        expected_type: ComponentType = ComponentType.NODE_GENERATOR

        result_id: str = generator.component_id
        result_type: ComponentType = generator.component_type

        self.assertEqual(expected_id, result_id)
        self.assertEqual(expected_type, result_type)



    def test_attributes_of_first_type_chebyshev_node_generator(self):
        generator: PipelineComponent = FirstTypeChebyshevNodeGenerator((-5.6142, 1.161), 13, int)

        expected_id: str = "Chebyshev1"
        expected_type: ComponentType = ComponentType.NODE_GENERATOR

        result_id: str = generator.component_id
        result_type: ComponentType = generator.component_type

        self.assertEqual(expected_id, result_id)
        self.assertEqual(expected_type, result_type)



    def test_attributes_of_second_type_chebyshev_node_generator(self):
        generator: PipelineComponent = SecondTypeChebyshevNodeGenerator((-556, -5), 1, int)

        expected_id: str = "Chebyshev2"
        expected_type: ComponentType = ComponentType.NODE_GENERATOR

        result_id: str = generator.component_id
        result_type: ComponentType = generator.component_type

        self.assertEqual(expected_id, result_id)
        self.assertEqual(expected_type, result_type)


if __name__ == '__main__':
    unittest.main()
