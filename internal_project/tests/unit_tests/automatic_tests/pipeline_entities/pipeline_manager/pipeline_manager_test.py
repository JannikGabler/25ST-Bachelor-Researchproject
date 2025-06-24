import unittest

from pipeline_entities.pipeline.pipeline import Pipeline
from pipeline_entities.pipeline_builder.pipeline_builder import PipelineBuilder
from pipeline_entities.pipeline_configuration.dataclasses.pipeline_configuration import PipelineConfiguration
from pipeline_entities.pipeline_configuration.dataclasses.pipeline_configuration_data import PipelineConfigurationData
from pipeline_entities.pipeline_input.pipeline_input import PipelineInput
from pipeline_entities.pipeline_input.pipeline_input_data import PipelineInputData
from pipeline_entities.pipeline_manager.pipeline_manager import PipelineManager
from setup_manager.internal_logic_setup_manager import InternalLogicSetupManager


class MyTestCase(unittest.TestCase):
    def setUp(self):
        InternalLogicSetupManager.setup()



    def test_only_dummy_components(self):
        pipeline_configuration_data: PipelineConfigurationData = PipelineConfigurationData()
        pipeline_configuration_data.supported_program_version = "Version(\"1.0.0\")"
        pipeline_configuration_data.components = "Tree(\"\"\"dummy\n dummy\n  dummy\"\"\")"

        pipeline_configuration: PipelineConfiguration = PipelineConfiguration(pipeline_configuration_data)

        pipeline_input_data: PipelineInputData = PipelineInputData()
        pipeline_input_data.data_type = "jax.numpy.float32"
        pipeline_input_data.node_count = "5"
        pipeline_input_data.interpolation_interval = "jax.numpy.array([-1, 1])"

        pipeline_input: PipelineInput = PipelineInput(pipeline_input_data)

        pipeline: Pipeline = PipelineBuilder.build(pipeline_configuration, pipeline_input)
        pipeline_manager: PipelineManager = PipelineManager(pipeline)

        pipeline_manager.execute_all()

        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
