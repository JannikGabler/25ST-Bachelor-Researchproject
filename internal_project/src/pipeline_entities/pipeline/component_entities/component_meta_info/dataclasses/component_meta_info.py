from dataclasses import dataclass

from pipeline_entities.pipeline.component_entities.constraints.abstracts.post_dynamic_constraint import (
    PostDynamicConstraint,
)
from pipeline_entities.pipeline.component_entities.constraints.abstracts.pre_dynamic_constraint import (
    PreDynamicConstraint,
)

# from pipeline_entities.constraints.pipeline_component.mixed_constraint import MixedConstraint
from pipeline_entities.pipeline.component_entities.constraints.abstracts.static_constraint import (
    StaticConstraint,
)

# ALLOW_ALL = object()


@dataclass(frozen=True)
class ComponentMetaInfo:
    attributes_modifying: set[str]

    attributes_allowed_to_be_overridden: set[str]  # | ALLOW_ALL # TODO

    pre_dynamic_constraints: list[PreDynamicConstraint]
    post_dynamic_constraints: list[PostDynamicConstraint]

    static_constraints: list[StaticConstraint]

    allow_multiple_executions_for_time_measurements: bool
    allow_additional_value_modifications_outside_specification: bool = False
