import unittest
import jax.numpy as jnp

from data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline.pipeline import Pipeline
from pipeline_entities.pipeline.pipeline_builder.pipeline_builder import PipelineBuilder
from data_classes.pipeline_configuration.pipeline_configuration import (
    PipelineConfiguration,
)
from data_classes.pipeline_configuration.pipeline_configuration_data import (
    PipelineConfigurationData,
)
from data_classes.pipeline_input.pipeline_input import PipelineInput
from data_classes.pipeline_input.pipeline_input_data import PipelineInputData
from pipeline_entities.pipeline_execution.pipeline_manager.pipeline_manager import (
    PipelineManager,
)
from setup_manager.internal_logic_setup_manager import InternalLogicSetupManager


class MyTestCase(unittest.TestCase):
    def setUp(self):
        InternalLogicSetupManager.setup()

    def test_only_dummy_components(self):
        pipeline_configuration_data: PipelineConfigurationData = (
            PipelineConfigurationData()
        )
        pipeline_configuration_data.supported_program_version = 'Version("1.0.0")'
        pipeline_configuration_data.components = 'Tree("""dummy\n dummy\n  dummy""")'

        pipeline_configuration: PipelineConfiguration = PipelineConfiguration(
            pipeline_configuration_data
        )

        pipeline_input_data: PipelineInputData = PipelineInputData()
        pipeline_input_data.data_type = "jax.numpy.float32"
        pipeline_input_data.node_count = "5"
        pipeline_input_data.interpolation_interval = "jax.numpy.array([-1, 1])"

        pipeline_input: PipelineInput = PipelineInput(pipeline_input_data)

        pipeline: Pipeline = PipelineBuilder.build(
            pipeline_configuration, pipeline_input
        )
        pipeline_manager: PipelineManager = PipelineManager(pipeline)

        pipeline_manager.execute_all()

        self.assertEqual(True, False)  # add assertion here

    def test_equidistant_node_generation_pipeline(self):
        pipeline_configuration_data: PipelineConfigurationData = (
            PipelineConfigurationData()
        )
        pipeline_configuration_data.supported_program_version = 'Version("1.0.0")'
        pipeline_configuration_data.components = (
            'Tree("""BaseInput\n Equidistant Node Generator\n""")'
        )

        pipeline_configuration: PipelineConfiguration = PipelineConfiguration(
            pipeline_configuration_data
        )

        pipeline_input_data: PipelineInputData = PipelineInputData()
        pipeline_input_data.data_type = "jax.numpy.float32"
        pipeline_input_data.node_count = "5"
        pipeline_input_data.interpolation_interval = "jax.numpy.array([-1, 1])"

        pipeline_input: PipelineInput = PipelineInput(pipeline_input_data)

        pipeline: Pipeline = PipelineBuilder.build(
            pipeline_configuration, pipeline_input
        )
        pipeline_manager: PipelineManager = PipelineManager(pipeline)

        pipeline_manager.execute_all()

        final_pipeline_data: PipelineData = pipeline_manager._pipeline_data_dict_["/0/"]

        self.assertEqual(jnp.float32, final_pipeline_data.data_type)
        self.assertEqual(5, final_pipeline_data.node_count)
        self.assertTrue(
            jnp.array_equal(
                jnp.array([-1, 1]), final_pipeline_data.interpolation_interval
            )
        )
        self.assertIsNone(final_pipeline_data.interpolation_values)

        self.assertTrue(
            jnp.array_equal(
                jnp.array([-1, -0.5, 0, 0.5, 1]), final_pipeline_data._nodes_
            )
        )

        self.assertEqual({}, final_pipeline_data.additional_values)


if __name__ == "__main__":
    unittest.main()
