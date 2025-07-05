from pipeline_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo
from pipeline_entities.constraints.default_constraints.default_static_constraints.attribute_required_constraint import \
    AttributeRequiredConstraint
from pipeline_entities.constraints.default_constraints.default_static_constraints.max_predecessors_constraint import \
    MaxPredecessorsConstraint
from pipeline_entities.constraints.default_constraints.default_static_constraints.min_predecessors_constraint import \
    MinPredecessorsConstraint

interpolation_values_evaluator_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"interpolation_values"},

    dynamic_constraints=[],

    static_constraints=[AttributeRequiredConstraint("data_type"), AttributeRequiredConstraint("function_callable"),
                        AttributeRequiredConstraint("interpolation_nodes"),
                        MinPredecessorsConstraint(1), MaxPredecessorsConstraint(1)],
)