from pipeline_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo
from pipeline_entities.constraints.default_constraints.default_dynamic_constraints.input_key_required_constraint import \
    InputKeyRequiredConstraint
from pipeline_entities.constraints.default_constraints.default_static_constraints.max_predecessors_constraint import \
    MaxPredecessorsConstraint
from pipeline_entities.constraints.default_constraints.default_static_constraints.min_predecessors_constraint import \
    MinPredecessorsConstraint

interpolation_values_input_component_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"interpolation_values"},

    dynamic_constraints=[InputKeyRequiredConstraint("interpolation_values")],

    static_constraints=[MinPredecessorsConstraint(1), MaxPredecessorsConstraint(1)],
)