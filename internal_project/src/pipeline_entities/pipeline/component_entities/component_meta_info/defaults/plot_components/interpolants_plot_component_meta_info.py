from pipeline_entities.pipeline.component_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.attribute_required_constraint import AttributeRequiredConstraint


"""
Component meta information for the interpolants plot component. 
This component does not allow any attributes to be overridden. 
It requires the attributes data_type, node_count, interpolation_interval, original_function, interpolation_nodes, 
interpolation_values and interpolant to be present in the pipeline data. 
In addition, it does not allow multiple executions for time measurements.
"""
interpolants_plot_component_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"plots"},

    attributes_allowed_to_be_overridden={"data_type"},

    pre_dynamic_constraints=[],

    post_dynamic_constraints=[],

    static_constraints=[AttributeRequiredConstraint("data_type"), AttributeRequiredConstraint("node_count"),
                        AttributeRequiredConstraint("interpolation_interval"), AttributeRequiredConstraint("original_function"),
                        AttributeRequiredConstraint("interpolation_nodes"), AttributeRequiredConstraint("interpolation_values"),
                        AttributeRequiredConstraint("interpolant")],

    allow_multiple_executions_for_time_measurements=False)
