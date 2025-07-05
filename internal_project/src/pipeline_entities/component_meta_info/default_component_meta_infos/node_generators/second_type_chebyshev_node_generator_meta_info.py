from pipeline_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo
from pipeline_entities.constraints.default_constraints.default_static_constraints.attribute_required_constraint import \
    AttributeRequiredConstraint

second_type_chebyshev_node_generator_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"nodes"},

    dynamic_constraints=[],

    static_constraints=[AttributeRequiredConstraint("data_type"), AttributeRequiredConstraint("node_count"),
                        AttributeRequiredConstraint("interpolation_interval")],

    #mixed_constraints=[]
)

