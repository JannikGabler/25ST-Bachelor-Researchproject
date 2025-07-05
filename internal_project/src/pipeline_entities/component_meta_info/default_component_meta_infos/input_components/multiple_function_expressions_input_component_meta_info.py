from pipeline_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo
from pipeline_entities.constraints.default_constraints.default_dynamic_constraints.input_key_required_constraint import \
    InputKeyRequiredConstraint
from pipeline_entities.constraints.default_constraints.default_static_constraints.attribute_required_constraint import \
    AttributeRequiredConstraint

multiple_function_expressions_input_component_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"function_values"},

    dynamic_constraints=[InputKeyRequiredConstraint("multiple_function_expressions"),
                         InputKeyRequiredConstraint("sympy_function_expression_simplification")],

    static_constraints=[AttributeRequiredConstraint("data_type"), AttributeRequiredConstraint("node_count"),
                        AttributeRequiredConstraint("nodes")],
    
    #mixed_constraints=[]
)
