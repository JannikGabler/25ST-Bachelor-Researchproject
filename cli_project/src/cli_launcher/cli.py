import argparse
import sys
import threading
import queue
import time
from pathlib import Path

from setup_manager.internal_logic_setup_manager import InternalLogicSetupManager
from file_handling.pipeline_input_handling.pipeline_input_file_manager import PipelineInputFileManager
from pipeline_entities.pipeline_configuration.dataclasses.pipeline_configuration_data import PipelineConfigurationData
from file_handling.pipeline_configuration_handling.pipeline_configuration_file_manager import PipelineConfigurationFileManager
from pipeline_entities.pipeline_configuration.dataclasses.pipeline_configuration import PipelineConfiguration
from pipeline_entities.pipeline_input.pipeline_input import PipelineInput
from pipeline_entities.pipeline_input.pipeline_input_data import PipelineInputData
from pipeline_entities.pipeline.pipeline import Pipeline
from pipeline_entities.pipeline_builder.pipeline_builder import PipelineBuilder
from pipeline_entities.pipeline_manager.pipeline_manager import PipelineManager
from cli_launcher.reporting import format_all_reports
from file_handling.dynamic_module_loading.dynamic_module_loader import DynamicModuleLoader

def _cli_worker(out_q: "queue.Queue[tuple[PipelineConfigurationData, PipelineInputData]]"):
    """
    Parses args, shows trust warning, loads the .ini files, then puts the parsed data on out_q.
    """
    parser = argparse.ArgumentParser(
        prog="interpolation_pipeline",
        description="Build and execute your interpolation pipeline"
    )

    ##################################################################
    # Option 1: Pipeline config and input files passed in seperately #
    ##################################################################
    parser.add_argument(
        "-pc", "--pipeline-config",
        help="Path to pipeline configuration file (.ini)"
    )
    parser.add_argument(
        "-pi", "--pipeline-input",
        help="Path to pipeline input file (.ini)"
    )

    #####################################################
    # Option 2: All pipeline files inside the directory #
    #####################################################
    parser.add_argument(
        "-d", "--directory",
        help="Folder containing pipeline config and input files (.ini)"
    )

    # Skip the arbitrary code execution trust warning
    parser.add_argument(
        "--skip-trust-warning", action="store_true",
        help="Skip the warning prompt for trusting the files"
    )
    args = parser.parse_args()

    if args.directory and (args.pipeline_config or args.pipeline_input):
        print("Cannot combine --directory with --pipeline-config/--pipeline-input", file=sys.stderr)
        sys.exit(1)

    # If neither files nor directory provided, default to current working directory
    if not args.directory and not args.pipeline_config and not args.pipeline_input:
        args.directory = str(Path.cwd())

    pcd_path = None
    pid_path = None
    if args.directory:
        folder = Path(args.directory)
        if not folder.is_dir():
            print(f"Not a directory: {folder}", file=sys.stderr)
            sys.exit(1)
        ini_files = list(folder.glob("*.ini"))
        for f in ini_files:
            content = f.read_text()
            if "[Pipeline]" in content:
                if pcd_path:
                    print(f"Duplicate pipeline config in folder: {f}", file=sys.stderr)
                    sys.exit(1)
                pcd_path = f
            if "[PipelineInput]" in content:
                if pid_path:
                    print(f"Duplicate pipeline input in folder: {f}", file=sys.stderr)
                    sys.exit(1)
                pid_path = f
        if not pcd_path or not pid_path:
            print("Missing pipeline config or pipeline input in directory", file=sys.stderr)
            sys.exit(1)
    else:
        if not args.pipeline_config or not args.pipeline_input:
            print("Both --pipeline-config and --pipeline-input are required when not using --directory", file=sys.stderr)
            sys.exit(1)
        pcd_path = Path(args.pipeline_config)
        pid_path = Path(args.pipeline_input)


    if not args.skip_trust_warning:
        print(
            "Warning! Loading .ini files may execute arbitrary code. "
            "Only use pipeline config and pipeline input files from trusted sources!",
            file=sys.stderr
        )
        reply = input("Do you trust these files? Proceed [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborting.", file=sys.stderr)
            sys.exit(1)

    pcd: PipelineConfigurationData = PipelineConfigurationFileManager.load_from_file(pcd_path)
    pid:PipelineInputData = PipelineInputFileManager.load_from_file(pid_path)

    # send to main thread
    out_q.put((pcd, pid))

def main():
    print("setting up internal logic...")
    InternalLogicSetupManager.setup()

    # CLI thread
    config_queue: "queue.Queue[tuple[PipelineConfigurationData, PipelineInputData]]" = queue.Queue() # new (empty) queue
    t = threading.Thread(target=_cli_worker, args=(config_queue,), daemon=True)
    t.start()

    # wait for the CLI thread to finish loading configs
    pipeline_config_data, pipeline_input_data = config_queue.get()

    # DynamicModuleLoader.load_directory()

    start_time = time.perf_counter()

    # stays on main thread
    pipeline_config = PipelineConfiguration(pipeline_config_data)
    pipeline_input = PipelineInput(pipeline_input_data)
    pipeline: Pipeline = PipelineBuilder.build(pipeline_config, pipeline_input)
    manager = PipelineManager(pipeline)

    print("\nInterpolation Pipeline (Modules):")
    print(pipeline)

    print("executing pipeline…")
    manager.execute_all()
    end_time = time.perf_counter()

    print("pipeline execution finished!\n")
    print("Results:")
    print(format_all_reports(manager._component_execution_reports_.values()))
    print(f"Total time: {end_time - start_time:.3f}s")

if __name__ == "__main__":
    main()