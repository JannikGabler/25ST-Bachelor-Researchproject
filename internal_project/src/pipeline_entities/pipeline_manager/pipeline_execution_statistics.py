from dataclasses import dataclass, field


@dataclass
class PipelineExecutionStatistics:
    component_init_durations: dict[str, float] = field(default_factory=dict)
    component_execution_durations: dict[str, float] = field(default_factory=dict)

