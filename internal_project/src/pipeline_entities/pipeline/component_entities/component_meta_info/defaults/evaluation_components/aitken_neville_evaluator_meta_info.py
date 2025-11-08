from pipeline_entities.pipeline.component_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.post_dynamic_constraints.pipeline_data_dtype_required_post_constraint import PipelineDataDtypeRequiredPostConstraint
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.attribute_required_constraint import AttributeRequiredConstraint
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.max_predecessors_constraint import MaxPredecessorsConstraint
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.min_predecessors_constraint import MinPredecessorsConstraint


"""
Component meta information for the Aitken–Neville evaluator. 
This component modifies the attribute interpolant_values and allows overriding of the attributes interpolant_evaluation_points, interpolant and data_type. 
It has no pre-dynamic constraints. A post-dynamic constraint ensures that interpolant_values is a jax.numpy array with the correct dtype. 
Furthermore, it requires the attributes data_type, node_count, interpolation_nodes, interpolation_values and interpolant_evaluation_points to be present in 
the pipeline data and enforces that the component has exactly one predecessor. 
Multiple executions for time measurements are allowed.
"""
aitken_neville_evaluator_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"interpolant_values"},

    attributes_allowed_to_be_overridden={"interpolant_evaluation_points", "interpolant", "data_type"},

    pre_dynamic_constraints=[],

    post_dynamic_constraints=[PipelineDataDtypeRequiredPostConstraint("interpolant_values")],

    static_constraints=[AttributeRequiredConstraint("data_type"), AttributeRequiredConstraint("node_count"),
                        AttributeRequiredConstraint("interpolation_nodes"), AttributeRequiredConstraint("interpolation_values"),
                        AttributeRequiredConstraint("interpolant_evaluation_points"), MinPredecessorsConstraint(1),
                        MaxPredecessorsConstraint(1)],

    allow_multiple_executions_for_time_measurements=True)
