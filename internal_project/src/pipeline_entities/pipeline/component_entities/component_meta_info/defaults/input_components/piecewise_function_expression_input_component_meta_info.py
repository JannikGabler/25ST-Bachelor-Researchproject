from pipeline_entities.pipeline.component_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.default_dynamic_constraints.input_key_required_constraint import \
    InputKeyRequiredConstraint
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.default_static_constraints.max_predecessors_constraint import \
    MaxPredecessorsConstraint
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.default_static_constraints.min_predecessors_constraint import \
    MinPredecessorsConstraint

piecewise_function_expression_input_component_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"function_callable"},

    dynamic_constraints=[InputKeyRequiredConstraint("piecewise_function_expression"),
                         InputKeyRequiredConstraint("sympy_function_expression_simplification")],

    static_constraints=[MinPredecessorsConstraint(1), MaxPredecessorsConstraint(1),],
)
