from pipeline_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo
from pipeline_entities.constraints.default_constraints.default_dynamic_constraints.input_key_required_constraint import \
    InputKeyRequiredConstraint

function_value_input_component_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"function_values"},

    dynamic_constraints=[InputKeyRequiredConstraint("function_values")],

    static_constraints=[],

    #mixed_constraints=[]
)