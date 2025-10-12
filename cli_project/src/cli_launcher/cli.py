import os
import sys

from argparse import ArgumentParser, Namespace
from pathlib import Path

from cli_launcher.reporting import format_all_reports
from constants.cli_project_constants import CLIConstants
from file_handling.pipeline_configuration_handling.pipeline_configuration_file_manager import (
    PipelineConfigurationFileManager,
)
from file_handling.result_persistence.filesystem_result_store import (
    FilesystemResultStore,
)
from file_handling.result_persistence.save_policy import SavePolicy
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
from data_classes.plot_template.plot_template import PlotTemplate
from pipeline_entities.pipeline.pipeline import Pipeline
from pipeline_entities.pipeline.pipeline_builder.pipeline_builder import PipelineBuilder
from pipeline_entities.pipeline_execution.output.pipeline_component_execution_report import (
    PipelineComponentExecutionReport,
)
from pipeline_entities.pipeline_execution.pipeline_manager.pipeline_manager import (
    PipelineManager,
)
from setup_manager.internal_logic_setup_manager import InternalLogicSetupManager
from utilities.rich_utilities import RichUtilities
from utilities.user_output_utilities import UserOutputUtilities


class CLI:
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
        self.skip_trust_warning = False

    def start(self) -> None:
        argument_namespace: Namespace = self.arg_parser.parse_args()
        self._parse_argument_namespace_(argument_namespace)
        self._print_argument_parsing_success_()
        self._perform_security_prompt_()
        self._setup_internal_logic_()
        pc, pi = self._parse_input_files_()

        pipeline: Pipeline = self._build_pipeline_(pc, pi)
        pipeline_manager: PipelineManager = self._execute_pipeline_(pipeline)
        self._print_results_(pipeline_manager)
        self._store_results(pipeline_manager)

        RichUtilities.close_panel()

        if CLIConstants.WAIT_FOR_ENTER_TO_EXIT:
            input("Press enter to exit.")

    @classmethod
    def _create_argument_parser_(cls) -> ArgumentParser:
        """
        Parses args, shows trust warning, loads the .ini files, then puts the parsed data on out_q.
        """
        return ArgumentParser(
            prog=CLIConstants.PROGRAM_NAME, description=CLIConstants.PROGRAM_DESCRIPTION
        )

    def _register_cli_parameters_(self) -> None:
        arg_parser: ArgumentParser = self.arg_parser

        self._register_cli_positional_arguments_()
        self._register_cli_file_parameters_group_()
        self._register_cli_directory_parameters_group_()

        # Skip the arbitrary code execution trust warning
        arg_parser.add_argument(
            "-st",
            "--skip-trust-warning",
            action="store_true",
            help="Skips the security warning prompt.",
        )

    def _register_cli_positional_arguments_(self) -> None:
        arg_parser: ArgumentParser = self.arg_parser

        arg_parser.add_argument(
            "directory",
            nargs="?",
            metavar="<PATH>",
            type=Path,
            help="Path to a pipeline directory.",
        )

    def _register_cli_file_parameters_group_(self) -> None:
        arg_parser: ArgumentParser = self.arg_parser

        group = arg_parser.add_argument_group(
            title="Pipeline configuration and -input files",
            description="Options for direct input of a pipeline configuration file and a pipeline input file without a "
            "pipeline directory.",
        )

        group.add_argument(
            "-pc",
            "--pipeline-config",
            metavar="<PATH>",
            type=Path,
            help="Path to a pipeline configuration file (.ini).",
        )
        group.add_argument(
            "-pi",
            "--pipeline-input",
            metavar="<PATH>",
            type=Path,
            help="Path to a pipeline input file (.ini).",
        )

    def _register_cli_directory_parameters_group_(self) -> None:
        arg_parser: ArgumentParser = self.arg_parser

        group = arg_parser.add_argument_group(
            title="Pipeline directory",
            description="Options for input of a pipeline directory. "
            "The pipeline directory must contain a pipeline configuration file (pipeline_configuration.ini) "
            "and a pipeline input file (pipeline_input.ini). It may also contain python files for dynamic "
            "loading.",
        )

        group.add_argument(
            "-d",
            "--directory",
            metavar="<PATH>",
            type=Path,
            help="Explicit option for the pipeline directory. "
            "Should be used mutually exclusive to the first position argument.",
        )

    def _parse_argument_namespace_(self, arg_namespace: Namespace) -> None:
        self.skip_trust_warning = arg_namespace.skip_trust_warning
        self._parse_directory_in_argument_namespace_(arg_namespace)
        self._parse_conf_file_in_argument_namespace_(arg_namespace)
        self._parse_input_file_in_argument_namespace_(arg_namespace)

    def _parse_directory_in_argument_namespace_(self, arg_namespace: Namespace) -> None:
        self.directory = (
            Path.cwd() if arg_namespace.directory is None else arg_namespace.directory
        )

        if not self.directory.exists():
            os.mkdir(self.directory)
            UserOutputUtilities.print_text(
                f"Specified pipeline directory {repr(self.directory)} could not be found. Creating it."
            )

        if not self.directory.is_dir():
            self.arg_parser.error(
                f"Specified path for the pipeline directory is no directory: {self.directory}."
            )

    def _parse_conf_file_in_argument_namespace_(self, arg_namespace: Namespace) -> None:
        if arg_namespace.pipeline_config is None:
            self.pipeline_config_file = self.directory / "pipeline_configuration.ini"
        else:
            self.pipeline_config_file = arg_namespace.pipeline_config

        if not self.pipeline_config_file.exists():
            self.arg_parser.error(
                f"The provided pipeline config file does not exist: '{self.pipeline_config_file.absolute()}'"
            )

        if not self.pipeline_config_file.is_file():
            self.arg_parser.error(
                f"Specified path for the pipeline config file is no file: '{self.pipeline_config_file.absolute()}'"
            )

        # self.arg_parser.error(
        #     f"Insufficient arguments were provided. "
        #     "Either a pipeline directory or pipeline config and pipeline input files must be specified. "
        #     f"Execute '{self.arg_parser.prog} --help' for more information.")

    def _parse_input_file_in_argument_namespace_(
        self, arg_namespace: Namespace
    ) -> None:
        if arg_namespace.pipeline_input is None:
            self.pipeline_input_file = self.directory / "pipeline_input.ini"
        else:
            self.pipeline_input_file = arg_namespace.pipeline_input

        if not self.pipeline_input_file.is_file():
            self.arg_parser.error(
                f"Specified path for the pipeline input file is no file: '{self.pipeline_input_file.absolute()}'"
            )

        if not self.pipeline_input_file.exists():
            self.arg_parser.error(
                f"The specified pipeline input file does not exist: '{self.pipeline_input_file.absolute()}'"
            )

    def _print_argument_parsing_success_(self) -> None:
        RichUtilities.open_panel("Successfully parsed arguments")
        RichUtilities.write_lines_in_panel(
            f"Pipeline directory: {self.directory.absolute()}"
        )
        RichUtilities.write_lines_in_panel(
            f"Pipeline config file: {self.pipeline_config_file.absolute()}"
        )
        RichUtilities.write_lines_in_panel(
            f"Pipeline input file: {self.pipeline_input_file.absolute()}"
        )

    def _perform_security_prompt_(self) -> None:
        if self.skip_trust_warning:
            return

        RichUtilities.open_panel("Security warning")
        RichUtilities.write_lines_in_panel(
            "Loading custom pipelines may execute arbitrary code. "
            "Only load files and directory from trusted sources!\n"
            "Would you like to proceed? [Y/n]",
            style="yellow",
        )

        RichUtilities.close_panel()

        if not RichUtilities.get_yes_no_input():
            sys.exit(0)

    @classmethod
    def _setup_internal_logic_(cls) -> None:
        RichUtilities.open_panel("Internal logic setup")
        RichUtilities.write_lines_in_panel("Setting up internal logic...")
        InternalLogicSetupManager.setup()
        RichUtilities.write_lines_in_panel("Successfully set up internal logic.")

    def _parse_input_files_(self) -> tuple[PipelineConfiguration, PipelineInput]:
        pc: PipelineConfiguration = self._parse_pipeline_configuration_file_()
        pi: PipelineInput = self._parse_pipeline_input_file_()
        return pc, pi

    def _parse_pipeline_configuration_file_(self) -> PipelineConfiguration:
        RichUtilities.open_panel("Loading pipeline configuration")
        RichUtilities.write_lines_in_panel("Loading pipeline configuration file:\n")

        path: Path = self.pipeline_config_file
        with open(path, "r", encoding="utf-8") as f:
            RichUtilities.write_lines_in_panel(f.read(), indent_level=1)

        RichUtilities.write_lines_in_panel(
            "\nParsing .ini format (PipelineConfigurationData)..."
        )

        pcd: PipelineConfigurationData = (
            PipelineConfigurationFileManager.load_from_file(path)
        )
        RichUtilities.write_lines_in_panel("Successfully parsed .ini format.")

        RichUtilities.write_lines_in_panel(
            "Parsing .ini format content (PipelineConfiguration)..."
        )
        pc: PipelineConfiguration = PipelineConfiguration(pcd)
        RichUtilities.write_lines_in_panel("Successfully parsed .ini format content.")

        return pc

    def _parse_pipeline_input_file_(self) -> PipelineInput:
        RichUtilities.open_panel("Loading pipeline input")
        RichUtilities.write_lines_in_panel("Loading pipeline input file:\n")

        path: Path = self.pipeline_input_file
        with open(path, "r", encoding="utf-8") as f:
            RichUtilities.write_lines_in_panel(f.read(), indent_level=1)

        RichUtilities.write_lines_in_panel(
            "\nParsing .ini format (PipelineInputData)..."
        )

        pid: PipelineInputData = PipelineInputFileManager.load_from_file(path)
        RichUtilities.write_lines_in_panel("Successfully parsed .ini format.")

        RichUtilities.write_lines_in_panel(
            "Parsing .ini format content (PipelineInput)..."
        )
        pi: PipelineInput = PipelineInput(pid)
        RichUtilities.write_lines_in_panel("Successfully parsed .ini format content.")

        return pi

    @classmethod
    def _build_pipeline_(cls, pc: PipelineConfiguration, pi: PipelineInput) -> Pipeline:
        RichUtilities.open_panel("Building pipeline")
        RichUtilities.write_lines_in_panel("Building pipeline...\n")

        pipeline: Pipeline = PipelineBuilder.build(pc, pi)
        RichUtilities.write_lines_in_panel(str(pipeline), indent_level=1)

        RichUtilities.write_lines_in_panel("\nSuccessfully built pipeline.")

        return pipeline

    @classmethod
    def _execute_pipeline_(cls, pipeline: Pipeline) -> PipelineManager:
        RichUtilities.open_panel("Executing pipeline")
        RichUtilities.write_lines_in_panel("Executing pipeline...")

        manager = PipelineManager(pipeline)
        manager.execute_all()
        RichUtilities.write_lines_in_panel("Successfully executed pipeline.")

        return manager

    @classmethod
    def _print_results_(cls, pipeline_manager: PipelineManager) -> None:
        RichUtilities.open_panel("Results")

        execution_reports: list[PipelineComponentExecutionReport] = (
            pipeline_manager.get_all_component_execution_reports()
        )
        output: str = format_all_reports(execution_reports)

        RichUtilities.write_lines_in_panel(output)

    @classmethod
    def _store_results(cls, pipeline_manager: PipelineManager) -> None:
        execution_reports: list[PipelineComponentExecutionReport] = (
            pipeline_manager.get_all_component_execution_reports()
        )

        policy = None  # TODO make this an optional user input

        store = FilesystemResultStore(output_root=Path.cwd() / "output")
        run_dir = store.save_run(execution_reports, policy)

        UserOutputUtilities.print_text(f"Saved results to: {run_dir}")
