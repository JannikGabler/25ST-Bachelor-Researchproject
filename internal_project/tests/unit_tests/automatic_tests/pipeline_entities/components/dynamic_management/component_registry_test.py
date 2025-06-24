import tempfile
import textwrap
import unittest
from pathlib import Path

from pipeline.components.default_components.default_node_generators.equidistant_node_generator import EquidistantNodeGenerator
from pipeline.components.default_components.default_node_generators.first_type_chebyshev_node_generator import \
    FirstTypeChebyshevNodeGenerator
from pipeline.components.default_components.default_node_generators.second_type_chebyshev_node_generator import \
    SecondTypeChebyshevNodeGenerator
from pipeline.components.dynamic_management.component_registry import ComponentRegistry
from pipeline.components.enums.component_type import ComponentType
from pipeline.component_infos.component_info.dataclasses.pipeline_component_info import PipelineComponentInfo


class MyTestCase(unittest.TestCase):

    def test_default_equidistant_node_generators(self):
        expected: PipelineComponentInfo = PipelineComponentInfo("Equidistant", ComponentType.NODE_GENERATOR, EquidistantNodeGenerator)

        result: PipelineComponentInfo = ComponentRegistry.get("Equidistant", ComponentType.NODE_GENERATOR)

        self.assertEqual(expected, result)



    def test_first_type_chebyshev_node_generators(self):
        expected: PipelineComponentInfo = PipelineComponentInfo("Chebyshev1", ComponentType.NODE_GENERATOR, FirstTypeChebyshevNodeGenerator)

        result: PipelineComponentInfo = ComponentRegistry.get("Chebyshev1", ComponentType.NODE_GENERATOR)

        self.assertEqual(expected, result)



    def test_second_type_chebyshev_node_generators(self):
        expected: PipelineComponentInfo = PipelineComponentInfo("Chebyshev2", ComponentType.NODE_GENERATOR, SecondTypeChebyshevNodeGenerator)

        result: PipelineComponentInfo = ComponentRegistry.get("Chebyshev2", ComponentType.NODE_GENERATOR)

        self.assertEqual(expected, result)



    def test_register_component_from_file_1(self):
        file_content: bytes = textwrap.dedent("""\
                                import jax
                                from pipeline.components.abstracts.node_generator import NodeGenerator
                                import jax.numpy as jnp
                                
                                from pipeline.components.abstracts.pipeline_component import pipeline_component
                                from pipeline.components.enums.component_type import ComponentType
                                
                                
                                @pipeline_component(id="Test1")
                                class EquidistantNodeGenerator(NodeGenerator):
                                
                                    @jax.jit
                                    def generate_nodes(self) -> jnp.ndarray:
                                        return jnp.linspace(self.__interval__[0], self.__interval__[1], self.__node_count__, dtype=self.__data_type__)
                                """).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            #print(f"Directory path = {temp_dir}")

            temp_file = Path(temp_dir + "/temp_file.py")

            with open(temp_file, "wb") as f:
                f.write(file_content)

            ComponentRegistry.register_component_from_file(temp_file)

            result: PipelineComponentInfo = ComponentRegistry.get("Test1", ComponentType.NODE_GENERATOR)

            self.assertIsNotNone(result)
            self.assertEqual("Test1", result.component_id)
            self.assertEqual(ComponentType.NODE_GENERATOR, result.component_type)



    def test_register_component_from_file_2(self):
        file_content: bytes = textwrap.dedent("""\
                                import jax
                                from pipeline.components.abstracts.node_generator import NodeGenerator
                                import jax.numpy as jnp

                                from pipeline.components.abstracts.pipeline_component import pipeline_component
                                from pipeline.components.enums.component_type import ComponentType


                                @pipeline_component(id="Test2")
                                class EquidistantNodeGenerator(NodeGenerator):

                                    @jax.jit
                                    def generate_nodes(self) -> jnp.ndarray:
                                        return jnp.linspace(self.__interval__[0], self.__interval__[1], self.__node_count__, dtype=self.__data_type__)
                                """).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            #print(f"Directory path = {temp_dir}")

            temp_file = Path(temp_dir + "/temp_file.py")

            with open(temp_file, "wb") as f:
                f.write(file_content)

            # This line is different compared to the previous test
            ComponentRegistry.register_component_from_file(str(temp_file))

            result: PipelineComponentInfo = ComponentRegistry.get("Test2", ComponentType.NODE_GENERATOR)

            self.assertIsNotNone(result)
            self.assertEqual("Test2", result.component_id)
            self.assertEqual(ComponentType.NODE_GENERATOR, result.component_type)


    def test_register_component_from_folder(self):
        file_content_1: bytes = textwrap.dedent("""\
                                import jax
                                from pipeline.components.abstracts.node_generator import NodeGenerator
                                import jax.numpy as jnp

                                from pipeline.components.abstracts.pipeline_component import pipeline_component
                                from pipeline.components.enums.component_type import ComponentType


                                @pipeline_component(id="Test31")
                                class EquidistantNodeGenerator(NodeGenerator):

                                    @jax.jit
                                    def generate_nodes(self) -> jnp.ndarray:
                                        return jnp.linspace(self.__interval__[0], self.__interval__[1], self.__node_count__, dtype=self.__data_type__)
                                """).encode("utf-8")

        file_content_2: bytes = textwrap.dedent("""\
                                import jax
                                from pipeline.components.abstracts.node_generator import NodeGenerator
                                import jax.numpy as jnp

                                from pipeline.components.abstracts.pipeline_component import pipeline_component
                                from pipeline.components.enums.component_type import ComponentType


                                @pipeline_component(id="Test32")
                                class EquidistantNodeGenerator(NodeGenerator):

                                    @jax.jit
                                    def generate_nodes(self) -> jnp.ndarray:
                                        return jnp.linspace(self.__interval__[0], self.__interval__[1], self.__node_count__, dtype=self.__data_type__)
                                """).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            #print(f"Directory path = {temp_dir}")

            temp_file_1 = Path(temp_dir + "/temp_file1.py")
            temp_file_2 = Path(temp_dir + "/temp_file2.py")

            with open(temp_file_1, "wb") as f:
                f.write(file_content_1)
            with open(temp_file_2, "wb") as f:
                f.write(file_content_2)

            ComponentRegistry.register_components_from_folder(temp_dir)

            result_1: PipelineComponentInfo = ComponentRegistry.get("Test31", ComponentType.NODE_GENERATOR)
            result_2: PipelineComponentInfo = ComponentRegistry.get("Test32", ComponentType.NODE_GENERATOR)

            self.assertIsNotNone(result_1)
            self.assertEqual("Test31", result_1.component_id)
            self.assertEqual(ComponentType.NODE_GENERATOR, result_1.component_type)

            self.assertIsNotNone(result_2)
            self.assertEqual("Test32", result_2.component_id)
            self.assertEqual(ComponentType.NODE_GENERATOR, result_2.component_type)


if __name__ == '__main__':
    unittest.main()
