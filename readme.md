# Interpolation Pipeline

## Set up the project

1. Clone this project
2. Create venv `python -m venv .venv` and activate `source .venv/bin/activate`
3. Install the requirements `pip install -r requirements.txt` *(This will install cli_project and internal_project in editable mode, as well as all other dependencies)*

## CLI usage

1. Create `pipeline_configuration.ini` and `pipeline_input.ini` files and note their filepaths
2. To run the pipeline, execute `interpolation_pipeline` with the following arguments:
    - **Option 1:** Specify pipeline configuration and input seperately.
        - `-pc <filepath>` or `--pipeline-configuration <filepath>` for the pipeline configuration
        - `-pi <filepath>` or `--pipeline-input <filepath>` for the pipeline input
        - **Important Note:** this approach does (currently) not support dynamically loading custom modules, so only built-in modules will work.

    - **Option 2:** Pass a directory containing all the files.
        - `-d <path>` or `--directory <path>` for the path of the directory containing the pipeline configuration and input files (.ini)

    - **Option 3:** Pass no arguments.
        Then the current working directory will be used and searched for configuration and input files, as well as custom modules.

**Example: `interpolation_pipeline -pc pipeline_configuration.ini -pi pipeline_input.ini`**

### Additional notes:

- When trying to load .ini files (pipeline configuration and input), you will be prompted with a warning and asked if you trust the authors of the files. This is because **arbitrary code can be run** through those input files. You can skip this step by using the argument `--skip-trust-warning`.