import os
import sys

from colorama import Fore
from argparse import ArgumentParser, Namespace
from pathlib import Path

from file_handling.pipeline_configuration_handling.pipeline_configuration_file_manager import \
    PipelineConfigurationFileManager
from file_handling.pipeline_input_handling.pipeline_input_file_manager import PipelineInputFileManager
from pipeline_entities.pipeline.pipeline import Pipeline
from pipeline_entities.pipeline_builder.pipeline_builder import PipelineBuilder
from pipeline_entities.pipeline_configuration.dataclasses.pipeline_configuration import PipelineConfiguration
from pipeline_entities.pipeline_configuration.dataclasses.pipeline_configuration_data import PipelineConfigurationData
from pipeline_entities.pipeline_input.pipeline_input import PipelineInput
from pipeline_entities.pipeline_input.pipeline_input_data import PipelineInputData
from pipeline_entities.pipeline_manager.pipeline_manager import PipelineManager
from setup_manager.internal_logic_setup_manager import InternalLogicSetupManager
from utilities.user_input_utilities import UserInputUtilities


class CLI:
    PROGRAM_NAME = "interpolation_pipeline"
    PROGRAM_DESCRIPTION = "Build and execute your interpolation pipeline."


    arg_parser: ArgumentParser

    directory: Path | None
    pipeline_config_file: Path | None
    pipeline_input_file: Path | None



    def __init__(self) -> None:
        self.arg_parser = self._create_argument_parser_()
        self._register_cli_parameters_()

        self.directory = None
        self.pipeline_config_file = None
        self.pipeline_input_file = None



    def start(self) -> None:
        argument_namespace: Namespace = self.arg_parser.parse_args()
        self._parse_argument_namespace_(argument_namespace)
        self._print_argument_parsing_success_()
        self._perform_security_prompt_()
        self._setup_internal_logic_()
        pipeline: Pipeline = self._parse_input_files_()
        #print(pipeline)

        manager = PipelineManager(pipeline)
        manager.execute_all()



    @classmethod
    def _create_argument_parser_(cls) -> ArgumentParser:
        """
        Parses args, shows trust warning, loads the .ini files, then puts the parsed data on out_q.
        """
        return ArgumentParser(prog=cls.PROGRAM_NAME, description=cls.PROGRAM_DESCRIPTION)



    def _register_cli_parameters_(self) -> None:
        arg_parser: ArgumentParser = self.arg_parser

        self._register_cli_positional_arguments_()
        self._register_cli_file_parameters_group_()
        self._register_cli_directory_parameters_group_()

        # Skip the arbitrary code execution trust warning
        arg_parser.add_argument(
            "--skip-trust-warning",
            action="store_true",
            help="Skips the security warning prompt."
        )



    def _register_cli_positional_arguments_(self) -> None:
        arg_parser: ArgumentParser = self.arg_parser

        arg_parser.add_argument(
            "directory",
            nargs="?",
            metavar="<PATH>",
            type=Path,
            help="Path to a pipeline directory."
        )



    def _register_cli_file_parameters_group_(self) -> None:
        arg_parser: ArgumentParser = self.arg_parser

        group = arg_parser.add_argument_group(title="Pipeline configuration and -input files",
            description="Options for direct input of a pipeline configuration file and a pipeline input file without a "
                        "pipeline directory.")

        group.add_argument(
            "-pc", "--pipeline-config",
            metavar="<PATH>",
            type=Path,
            help="Path to a pipeline configuration file (.ini)."
        )
        group.add_argument(
            "-pi", "--pipeline-input",
            metavar="<PATH>",
            type=Path,
            help="Path to a pipeline input file (.ini)."
        )



    def _register_cli_directory_parameters_group_(self) -> None:
        arg_parser: ArgumentParser = self.arg_parser

        group = arg_parser.add_argument_group(title="Pipeline directory",
            description="Options for input of a pipeline directory. "
                        "The pipeline directory must contain a pipeline configuration file (pipeline_configuration.ini) "
                        "and a pipeline input file (pipeline_input.ini). It may also contain python files for dynamic "
                        "loading.")


        group.add_argument(
            "-d", "--directory",
            metavar="<PATH>",
            type=Path,
            help="Explicit option for the pipeline directory. "
                 "Should be used mutually exclusive to the first position argument.",
        )



    def _parse_argument_namespace_(self, arg_namespace: Namespace) -> None:
        self._parse_directory_in_argument_namespace_(arg_namespace)
        self._parse_conf_file_in_argument_namespace_(arg_namespace)
        self._parse_input_file_in_argument_namespace_(arg_namespace)



    def _parse_directory_in_argument_namespace_(self, arg_namespace: Namespace) -> None:
        self.directory = Path.cwd() if arg_namespace.directory is None else arg_namespace.directory

        if not self.directory.exists():
            os.mkdir(self.directory)
            print(f"Specified pipeline directory {repr(self.directory)} could not be found. Creating it.")

        if not self.directory.is_dir():
            self.arg_parser.error(f"Specified path for the pipeline directory is no directory: {self.directory}.")



    def _parse_conf_file_in_argument_namespace_(self, arg_namespace: Namespace) -> None:
        if arg_namespace.pipeline_config is None:
            self.pipeline_config_file = self.directory / "pipeline_configuration.ini"
        else:
            self.pipeline_config_file = arg_namespace.pipeline_config

        if not self.pipeline_config_file.exists():
            self.arg_parser.error(f"The provided pipeline config file does not exist: '{self.pipeline_config_file.absolute()}'")

        if not self.pipeline_config_file.is_file():
            self.arg_parser.error(f"Specified path for the pipeline config file is no file: '{self.pipeline_config_file.absolute()}'")



        # self.arg_parser.error(
        #     f"Insufficient arguments were provided. "
        #     "Either a pipeline directory or pipeline config and pipeline input files must be specified. "
        #     f"Execute '{self.arg_parser.prog} --help' for more information.")



    def _parse_input_file_in_argument_namespace_(self, arg_namespace: Namespace) -> None:
        if arg_namespace.pipeline_input is None:
            self.pipeline_input_file = self.directory / "pipeline_input.ini"
        else:
            self.pipeline_input_file = arg_namespace.pipeline_input

        if not self.pipeline_input_file.is_file():
            self.arg_parser.error(
                f"Specified path for the pipeline input file is no file: '{self.pipeline_input_file.absolute()}'")

        if not self.pipeline_input_file.exists():
            self.arg_parser.error(f"The specified pipeline input file does not exist: '{self.pipeline_input_file.absolute()}'")



    def _print_argument_parsing_success_(self) -> None:
        print("")
        print("--- Successfully parsed arguments ---")
        print(f"Pipeline directory: {self.directory.absolute()}")
        print(f"Pipeline config file: {self.pipeline_config_file.absolute()}")
        print(f"Pipeline input file: {self.pipeline_input_file.absolute()}")



    @classmethod
    def _perform_security_prompt_(cls) -> None:
        print(Fore.YELLOW +
            "\n--- Security warning ---\n"
            "Loading custom pipelines may execute arbitrary code. "
            "Only load files and directory from trusted sources!\n"
            "Would you like to proceed? [y/n]"
            + Fore.RESET)

        if UserInputUtilities.get_yes_no_input() is False:
            sys.exit(0)



    @classmethod
    def _setup_internal_logic_(cls) -> None:
        print("\n--- Internal logic setup ---")
        print("Setting up internal logic...")
        InternalLogicSetupManager.setup()
        print("Successfully set up internal logic.")



    def _parse_input_files_(self) -> Pipeline:
        pc: PipelineConfiguration = self._parse_pipeline_configuration_file_()
        pi: PipelineInput = self._parse_pipeline_input_file_()
        return PipelineBuilder.build(pc, pi)



    def _parse_pipeline_configuration_file_(self) -> PipelineConfiguration:
        print("\n--- Loading pipeline configuration ---")
        print("Loading pipeline configuration file:\n")

        path: Path = self.pipeline_config_file
        with open(path, "r", encoding="utf-8") as f:
            print(f.read())

        print("\nParsing .ini format (PipelineConfigurationData)...")

        pcd: PipelineConfigurationData = PipelineConfigurationFileManager.load_from_file(path)
        print("Successfully parsed .ini format.")

        print("Parsing .ini format content (PipelineConfiguration)...")
        pc: PipelineConfiguration = PipelineConfiguration(pcd)
        print("Successfully parsed .ini format content.")

        return pc



    def _parse_pipeline_input_file_(self) -> PipelineInput:
        print("\n--- Loading pipeline input ---")
        print("Loading pipeline input file:\n")

        path: Path = self.pipeline_input_file
        with open(path, "r", encoding="utf-8") as f:
            print(f.read())

        print("\nParsing .ini format (PipelineInputData)...")

        pid: PipelineInputData = PipelineInputFileManager.load_from_file(path)
        print("Successfully parsed .ini format.")

        print("Parsing .ini format content (PipelineInput)...")
        pi: PipelineInput = PipelineInput(pid)
        print("Successfully parsed .ini format content.")

        return pi
