from dataclasses import dataclass, field


@dataclass
class PipelineConfigurationData:
    """
    Configuration container for pipeline execution. This dataclass provides general metadata such as pipeline name, supported program version,
    enabled components, runtime measurement settings, and optional additional values.

    Attributes:
        name (str | None): The name of the pipeline.
        supported_program_version (str | None): The program version that this configuration is compatible with.
        components (str | None): String specification of the components to be used in the pipeline.
        runs_for_component_execution_time_measurements (int | None): Number of runs to perform for measuring component execution times.
        additional_values (dict[str, str] | None): Optional dictionary of additional configuration values.
    """

    name: str | None = None
    supported_program_version: str | None = None

    components: str | None = None

    runs_for_component_execution_time_measurements: int | None = None

    additional_values: dict[str, str] | None = field(default_factory=dict)
