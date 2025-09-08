from pipeline_entities.pipeline.component_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.attribute_required_constraint import (
    AttributeRequiredConstraint,
)
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.max_predecessors_constraint import (
    MaxPredecessorsConstraint,
)
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.min_predecessors_constraint import (
    MinPredecessorsConstraint,
)

chebyshev_interpolation_matrix_core_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"interpolation_matrix"},

    attributes_allowed_to_be_overridden={"interpolation_nodes", "data_type", "node_count"},

    pre_dynamic_constraints=[],

    post_dynamic_constraints=[],

    static_constraints=[
        AttributeRequiredConstraint("interpolation_nodes"),
        AttributeRequiredConstraint("data_type"),
        AttributeRequiredConstraint("node_count"),
        MinPredecessorsConstraint(1),
        MaxPredecessorsConstraint(1),
    ],

    allow_multiple_executions_for_time_measurements=True,
)
