from pipeline_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo
from pipeline_entities.constraints.default_constraints.default_static_constraints.attribute_required_constraint import \
    AttributeRequiredConstraint
from pipeline_entities.constraints.default_constraints.default_static_constraints.max_predecessors_constraint import \
    MaxPredecessorsConstraint
from pipeline_entities.constraints.default_constraints.default_static_constraints.min_predecessors_constraint import \
    MinPredecessorsConstraint

aitken_neville_interpolation_core_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"interpolant"},

    dynamic_constraints=[],

    static_constraints=[AttributeRequiredConstraint("interpolation_nodes"), AttributeRequiredConstraint("interpolation_values"),
                        MinPredecessorsConstraint(1), MaxPredecessorsConstraint(1)],
)

