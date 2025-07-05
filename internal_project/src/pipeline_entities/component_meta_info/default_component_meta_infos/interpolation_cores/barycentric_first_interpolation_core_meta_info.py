from pipeline_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo
from pipeline_entities.constraints.default_constraints.default_static_constraints.attribute_required_constraint import \
    AttributeRequiredConstraint

barycentric_first_interpolation_core_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"interpolant"},

    dynamic_constraints=[],

    static_constraints=[AttributeRequiredConstraint("nodes")],

    #mixed_constraints=[]
)

