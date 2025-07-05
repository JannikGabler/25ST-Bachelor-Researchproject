from pipeline_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo
from pipeline_entities.constraints.default_constraints.default_static_constraints.attribute_required_constraint import \
    AttributeRequiredConstraint

plot_component_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying=set(),

    dynamic_constraints=[],

    static_constraints=[AttributeRequiredConstraint("interpolation_interval"), AttributeRequiredConstraint("function_callable"),
                        AttributeRequiredConstraint("nodes"), AttributeRequiredConstraint("interpolant")],

    mixed_constraints=[]
)

