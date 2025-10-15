from pipeline_entities.pipeline.component_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.post_dynamic_constraints.pipeline_data_dtype_required_post_constraint import PipelineDataDtypeRequiredPostConstraint
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.attribute_required_constraint import AttributeRequiredConstraint
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.max_predecessors_constraint import MaxPredecessorsConstraint
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.min_predecessors_constraint import MinPredecessorsConstraint


"""
Component meta information for the interpolant evaluator. 
This component modifies the attribute interpolant_values and allows overriding of the attributes interpolant_evaluation_points, interpolant, data_type and 
use_compensation. 
It has no pre-dynamic constraints. A post-dynamic constraint ensures that interpolant_values is a jax.numpy array with the correct dtype. 
Furthermore, it requires the attributes interpolant_evaluation_points, interpolant and data_type to be present in the pipeline data and enforces that the 
component has exactly one predecessor. 
Multiple executions for time measurements are allowed.
"""
interpolant_evaluator_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"interpolant_values"},

    attributes_allowed_to_be_overridden={"interpolant_evaluation_points", "interpolant", "data_type", "use_compensation"},

    pre_dynamic_constraints=[],

    post_dynamic_constraints=[PipelineDataDtypeRequiredPostConstraint("interpolant_values")],

    static_constraints=[AttributeRequiredConstraint("interpolant_evaluation_points"), AttributeRequiredConstraint("interpolant"),
                        AttributeRequiredConstraint("data_type"), MinPredecessorsConstraint(1), MaxPredecessorsConstraint(1)],

    allow_multiple_executions_for_time_measurements=True)

