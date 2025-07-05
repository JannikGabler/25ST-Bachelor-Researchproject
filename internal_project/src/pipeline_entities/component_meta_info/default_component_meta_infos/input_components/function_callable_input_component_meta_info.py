from pipeline_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo
from pipeline_entities.constraints.default_constraints.default_dynamic_constraints.input_key_required_constraint import \
    InputKeyRequiredConstraint
from pipeline_entities.constraints.default_constraints.default_static_constraints.attribute_required_constraint import \
    AttributeRequiredConstraint
from pipeline_entities.constraints.default_constraints.default_static_constraints.max_predecessors_constraint import \
    MaxPredecessorsConstraint
from pipeline_entities.constraints.default_constraints.default_static_constraints.min_predecessors_constraint import \
    MinPredecessorsConstraint

function_callable_input_component_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"function_callable"},

    dynamic_constraints=[InputKeyRequiredConstraint("function_callable")],

    static_constraints=[MinPredecessorsConstraint(1), MaxPredecessorsConstraint(1)],
)