from pipeline_entities.pipeline.component_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.attribute_required_constraint import AttributeRequiredConstraint


"""
Component meta information for the absolute error plot component.
This component allows overriding of the attributes y_limit, y_scale and y_scale_base.
It requires the attributes data_type, interpolation_interval, original_function and interpolant to be present in the pipeline data.
In addition, it does not allow multiple executions for time measurements.
"""
absolut_error_plot_component_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"plots"},

    attributes_allowed_to_be_overridden={"data_type", "y_limit", "y_scale", "y_scale_base"},

    pre_dynamic_constraints=[],

    post_dynamic_constraints=[],

    static_constraints=[AttributeRequiredConstraint("data_type"), AttributeRequiredConstraint("interpolation_interval"),
                        AttributeRequiredConstraint("original_function"), AttributeRequiredConstraint("interpolant")],

    allow_multiple_executions_for_time_measurements=False)
