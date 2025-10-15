from pipeline_entities.pipeline.component_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.pre_dynamic_constraints.input_key_required_constraint import InputKeyRequiredConstraint


"""
Component meta information for the base input pipeline component. 
This component modifies the attributes data_type, node_count, interpolation_interval and interpolant_evaluation_points and allows all of them to be overridden. 
It has pre-dynamic constraints that require the keys data_type, node_count, interpolation_interval and interpolant_evaluation_points to be provided in the 
pipeline input, while it has no post-dynamic or static constraints. 
Multiple executions for time measurements are not allowed, but additional value modifications outside the specification are permitted.
"""
base_input_pipeline_component_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"data_type", "node_count", "interpolation_interval", "interpolant_evaluation_points"},

    attributes_allowed_to_be_overridden={"data_type", "node_count", "interpolation_interval", "interpolant_evaluation_points"},

    pre_dynamic_constraints=[InputKeyRequiredConstraint("data_type"), InputKeyRequiredConstraint("node_count"),
                             InputKeyRequiredConstraint("interpolation_interval"), InputKeyRequiredConstraint("interpolant_evaluation_points")],

    post_dynamic_constraints=[],

    static_constraints=[],

    allow_multiple_executions_for_time_measurements=False,

    allow_additional_value_modifications_outside_specification=True)
