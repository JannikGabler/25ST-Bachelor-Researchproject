from pipeline_entities.pipeline.component_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.post_dynamic_constraints.pipeline_data_dtype_required_post_constraint import PipelineDataDtypeRequiredPostConstraint
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.attribute_required_constraint import AttributeRequiredConstraint
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.max_predecessors_constraint import MaxPredecessorsConstraint
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.min_predecessors_constraint import MinPredecessorsConstraint


"""
Component meta information for the second type Chebyshev node generator. 
This component modifies the attribute interpolation_nodes. 
It allows overriding of the attributes data_type, node_count and interpolation_interval.
A post-dynamic constraint ensures that interpolation_nodes is a jax.numpy array with the correct dtype. 
Furthermore, it requires the attributes data_type, node_count and interpolation_interval to be present in the pipeline data and it enforces 
that the component has exactly one predecessor. 
Multiple executions for time measurements are allowed.
"""
second_type_chebyshev_node_generator_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"interpolation_nodes"},

    attributes_allowed_to_be_overridden={"data_type", "node_count", "interpolation_interval"},

    pre_dynamic_constraints=[],

    post_dynamic_constraints=[PipelineDataDtypeRequiredPostConstraint("interpolation_nodes")],

    static_constraints=[AttributeRequiredConstraint("data_type"), AttributeRequiredConstraint("node_count"),
                        AttributeRequiredConstraint("interpolation_interval"), MinPredecessorsConstraint(1), MaxPredecessorsConstraint(1)],

    allow_multiple_executions_for_time_measurements=True,
)
