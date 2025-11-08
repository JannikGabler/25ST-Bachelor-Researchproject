from dataclasses import dataclass
from pipeline_entities.pipeline.component_entities.constraints.abstracts.post_dynamic_constraint import PostDynamicConstraint
from pipeline_entities.pipeline.component_entities.constraints.abstracts.pre_dynamic_constraint import PreDynamicConstraint
from pipeline_entities.pipeline.component_entities.constraints.abstracts.static_constraint import StaticConstraint


@dataclass(frozen=True)
class ComponentMetaInfo:
    """
    Meta information for a pipeline component.
    This dataclass specifies which attributes a component modifies, which attributes may be overridden, and which constraints must hold.
    It also defines whether multiple executions are allowed for time measurements and whether additional value modifications outside the
    specification are permitted.

    Attributes:
        attributes_modifying (set[str]): Attributes of PipelineData that are modified by the component.
        attributes_allowed_to_be_overridden (set[str]): Attributes that may be overridden.
        pre_dynamic_constraints (list[PreDynamicConstraint]): Constraints checked before component execution using the current pipeline data and input.
        post_dynamic_constraints (list[PostDynamicConstraint]): Constraints checked after component execution using input and output data.
        static_constraints (list[StaticConstraint]): Constraints checked statically on the pipeline graph structure.
        allow_multiple_executions_for_time_measurements (bool): Whether the component may be executed multiple times for timing purposes.
        allow_additional_value_modifications_outside_specification (bool): Whether modifications of additional values not listed in the specification are
        permitted.
    """

    attributes_modifying: set[str]

    attributes_allowed_to_be_overridden: set[str]  # | ALLOW_ALL # TODO

    pre_dynamic_constraints: list[PreDynamicConstraint]
    post_dynamic_constraints: list[PostDynamicConstraint]

    static_constraints: list[StaticConstraint]

    allow_multiple_executions_for_time_measurements: bool
    allow_additional_value_modifications_outside_specification: bool = False
