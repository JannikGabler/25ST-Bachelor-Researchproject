from dataclasses import dataclass

from pipeline_entities.constraints.abstracts.dynamic_constraint import DynamicConstraint
#from pipeline_entities.constraints.abstracts.mixed_constraint import MixedConstraint
from pipeline_entities.constraints.abstracts.static_constraint import StaticConstraint


@dataclass(frozen=True)
class ComponentMetaInfo:
    attributes_modifying: set[str]

    dynamic_constraints: list[DynamicConstraint]
    static_constraints: list[StaticConstraint]
    #mixed_constraints: list[MixedConstraint]

    allow_additional_value_modifications_outside_specification: bool = False

