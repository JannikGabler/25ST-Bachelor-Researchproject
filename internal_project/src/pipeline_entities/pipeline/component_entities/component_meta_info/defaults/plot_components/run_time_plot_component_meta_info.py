from pipeline_entities.pipeline.component_entities.component_meta_info.dataclasses.component_meta_info import (
    ComponentMetaInfo,
)
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.attribute_required_constraint import (
    AttributeRequiredConstraint,
)

run_time_component_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"plots"},
    attributes_allowed_to_be_overridden=set(),
    pre_dynamic_constraints=[],
    post_dynamic_constraints=[],
    static_constraints=[
        AttributeRequiredConstraint("interpolation_interval"),
        AttributeRequiredConstraint("original_function"),
        AttributeRequiredConstraint("interpolation_nodes"),
        AttributeRequiredConstraint("interpolant"),
    ],
    allow_multiple_executions_for_time_measurements=False,
)
