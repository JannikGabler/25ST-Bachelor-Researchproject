import tempfile
import textwrap
from pathlib import Path

from file_handling.pipeline_configuration_handling.pipeline_configuration_file_manager import (
    PipelineConfigurationFileManager,
)
from file_handling.pipeline_input_handling.pipeline_input_file_manager import (
    PipelineInputFileManager,
)
from data_classes.pipeline_configuration.pipeline_configuration import (
    PipelineConfiguration,
)
from data_classes.pipeline_configuration.pipeline_configuration_data import (
    PipelineConfigurationData,
)
from data_classes.pipeline_input.pipeline_input import PipelineInput
from data_classes.pipeline_input.pipeline_input_data import PipelineInputData
from pipeline_entities.pipeline.pipeline import Pipeline
from pipeline_entities.pipeline.pipeline_builder.pipeline_builder import PipelineBuilder
from pipeline_entities.pipeline_execution.pipeline_manager.pipeline_manager import (
    PipelineManager,
)
from setup_manager.internal_logic_setup_manager import InternalLogicSetupManager

InternalLogicSetupManager.setup()

temp_pipeline_configuration_file = Path(
    "C:/Users/janni/DEV/Projects/University/6. Semester/25ST-Bachelor-Researchproject/research/src/research_question_3/sin_function/equidistant/barycentric1/bfloat16/pipeline_configuration.ini"
)

pipeline_configuration_data: PipelineConfigurationData = (
    PipelineConfigurationFileManager.load_from_file(temp_pipeline_configuration_file)
)
# print(pipeline_configuration_data)

pipeline_configuration: PipelineConfiguration = PipelineConfiguration(
    pipeline_configuration_data
)
# print(pipeline_configuration)

temp_pipeline_input_file = Path(
    "C:/Users/janni/DEV/Projects/University/6. Semester/25ST-Bachelor-Researchproject/research/src/research_question_3/sin_function/equidistant/barycentric1/bfloat16/pipeline_input.ini"
)

pipeline_input_data: PipelineInputData = PipelineInputFileManager.load_from_file(
    temp_pipeline_input_file
)
# print(pipeline_input_data)

pipeline_input: PipelineInput = PipelineInput(pipeline_input_data)
# print(pipeline_input)


pipeline: Pipeline = PipelineBuilder.build(pipeline_configuration, pipeline_input)

pipeline_manager: PipelineManager = PipelineManager(pipeline)
pipeline_manager.execute_all()
