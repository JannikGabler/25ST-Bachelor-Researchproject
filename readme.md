# Interpolation Pipeline

## CLI usage

1. Install the local cli package with `pip install -e cli_project` *(editable mode while in development)*
2. Create `pipeline_configuration.ini` and `pipeline_input.ini` files and note their filepaths
3. To run the pipeline, execute `interpolation_pipeline` with the following arguments:
    - `-pc <filepath>` or `--pipeline-configuration <filepath>` for the pipeline configuration
    - `-pi <filepath>` or `--pipeline-input <filepath>` for the pipeline input

**Example: `interpolation_pipeline -pc pipeline_configuration.ini -pi pipeline_input.ini`**