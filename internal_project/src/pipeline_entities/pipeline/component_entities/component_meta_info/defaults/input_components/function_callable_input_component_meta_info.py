from pipeline_entities.pipeline.component_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.pre_dynamic_constraints.input_key_required_constraint import \
    InputKeyRequiredConstraint
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.max_predecessors_constraint import \
    MaxPredecessorsConstraint
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.min_predecessors_constraint import \
    MinPredecessorsConstraint

function_callable_input_component_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"function_callable"},

    attributes_allowed_to_be_overridden={"function_callable"},

    pre_dynamic_constraints=[InputKeyRequiredConstraint("function_callable")],

    post_dynamic_constraints=[],

    static_constraints=[MinPredecessorsConstraint(1), MaxPredecessorsConstraint(1)],
)