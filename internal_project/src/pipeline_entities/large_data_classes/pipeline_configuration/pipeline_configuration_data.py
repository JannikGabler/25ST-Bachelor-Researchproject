from dataclasses import dataclass, field


@dataclass
class PipelineConfigurationData:
    name: str | None = None
    supported_program_version: str | None = None

    components: str | None = None

    additional_values: dict[str, str] | None = field(default_factory=dict)

