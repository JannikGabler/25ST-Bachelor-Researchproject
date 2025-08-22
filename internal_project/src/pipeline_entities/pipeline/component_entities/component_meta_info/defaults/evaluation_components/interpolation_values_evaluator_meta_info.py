from pipeline_entities.pipeline.component_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.post_dynamic_constraints.pipeline_data_dtype_required_post_constraint import \
    PipelineDataDtypeRequiredPostConstraint
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.attribute_required_constraint import \
    AttributeRequiredConstraint
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.max_predecessors_constraint import \
    MaxPredecessorsConstraint
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.min_predecessors_constraint import \
    MinPredecessorsConstraint


interpolation_values_evaluator_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"interpolation_values"},

    attributes_allowed_to_be_overridden={"data_type", "original_function", "interpolation_nodes"},

    pre_dynamic_constraints=[],

    post_dynamic_constraints=[PipelineDataDtypeRequiredPostConstraint("interpolation_values")],

    static_constraints=[AttributeRequiredConstraint("data_type"), AttributeRequiredConstraint("original_function"),
                        AttributeRequiredConstraint("interpolation_nodes"),
                        MinPredecessorsConstraint(1), MaxPredecessorsConstraint(1)],
)