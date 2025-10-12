from pipeline_entities.pipeline.component_entities.component_meta_info.dataclasses.component_meta_info import (
    ComponentMetaInfo,
)
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.pre_dynamic_constraints.input_key_required_constraint import (
    InputKeyRequiredConstraint,
)

base_input_pipeline_component_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={
        "data_type",
        "node_count",
        "interpolation_interval",
        "interpolant_evaluation_points",
    },
    attributes_allowed_to_be_overridden={
        "data_type",
        "node_count",
        "interpolation_interval",
        "interpolant_evaluation_points",
    },
    pre_dynamic_constraints=[
        InputKeyRequiredConstraint("data_type"),
        InputKeyRequiredConstraint("node_count"),
        InputKeyRequiredConstraint("interpolation_interval"),
        InputKeyRequiredConstraint("interpolant_evaluation_points"),
    ],
    post_dynamic_constraints=[],
    static_constraints=[],
    allow_multiple_executions_for_time_measurements=False,
    allow_additional_value_modifications_outside_specification=True,
)
