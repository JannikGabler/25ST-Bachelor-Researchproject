# cli_project/src/cli_project/cli.py
import argparse
import sys

from pathlib import Path

from file_handling.pipeline_input_handling.pipeline_input_file_manager import PipelineInputFileManager
from pipeline_entities.pipeline_configuration.dataclasses.pipeline_configuration_data import PipelineConfigurationData
from file_handling.pipeline_configuration_handling.pipeline_configuration_file_manager import \
    PipelineConfigurationFileManager
from pipeline_entities.pipeline_configuration.dataclasses.pipeline_configuration import PipelineConfiguration
from pipeline_entities.pipeline_input.pipeline_input import PipelineInput
from pipeline_entities.pipeline_input.pipeline_input_data import PipelineInputData
from pipeline_entities.pipeline.pipeline import Pipeline
from pipeline_entities.pipeline_builder.pipeline_builder import PipelineBuilder
from pipeline_entities.pipeline_manager.pipeline_manager import PipelineManager
from setup_manager.internal_logic_setup_manager import InternalLogicSetupManager

def main():
    print("setting up internal logic...")
    InternalLogicSetupManager.setup()

    print("reading pipeline config and input files...")

    parser = argparse.ArgumentParser(
        prog="interpolation_pipeline",
        description="Build and execute your interpolation pipeline"
    )
    parser.add_argument(
        "-pc", "--pipeline-config", required=True,
        help="Path to pipeline configuration file (.ini)"
    )
    parser.add_argument(
        "-pi", "--pipeline-input", required=True,
        help="Path to pipeline input file (.ini)"
    )
    args = parser.parse_args()

    print("parsed arguments:", args)

    try:
        pcd_path = Path(args.pipeline_config)
        pipeline_configuration_data: PipelineConfigurationData = PipelineConfigurationFileManager.load_from_file(pcd_path)
        pipeline_configuration: PipelineConfiguration = PipelineConfiguration(pipeline_configuration_data)

        pid_path = Path(args.pipeline_input)
        pipeline_input_data: PipelineInputData = PipelineInputFileManager.load_from_file(pid_path)
        pipeline_input: PipelineInput = PipelineInput(pipeline_input_data)
        
        pipeline: Pipeline = PipelineBuilder.build(pipeline_configuration, pipeline_input)

        manager  = PipelineManager(pipeline)
        manager.execute_all()
        print(manager._pipeline_data_dict_["/0/0/0/"])
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise e
        # sys.exit(1)