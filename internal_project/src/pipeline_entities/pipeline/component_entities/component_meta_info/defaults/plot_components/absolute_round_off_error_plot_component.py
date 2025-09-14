from constants.internal_logic_constants import AbsoluteRoundOffErrorPlotComponentConstants
from pipeline_entities.pipeline.component_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.attribute_required_constraint import \
    AttributeRequiredConstraint


absolut_round_off_error_plot_component_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"plots"},

    attributes_allowed_to_be_overridden={AbsoluteRoundOffErrorPlotComponentConstants.Y_THRESHOLD_ATTRIBUTE_NAME,
                                         AbsoluteRoundOffErrorPlotComponentConstants.Y_LIMIT_ATTRIBUTE_NAME},

    pre_dynamic_constraints=[],

    post_dynamic_constraints=[],

    static_constraints=[AttributeRequiredConstraint("data_type"),
                        AttributeRequiredConstraint("node_count"),
                        AttributeRequiredConstraint("interpolation_interval"),
                        AttributeRequiredConstraint("interpolation_nodes"),
                        AttributeRequiredConstraint("interpolation_values"),
                        AttributeRequiredConstraint("interpolant")],

    allow_multiple_executions_for_time_measurements=False
)


