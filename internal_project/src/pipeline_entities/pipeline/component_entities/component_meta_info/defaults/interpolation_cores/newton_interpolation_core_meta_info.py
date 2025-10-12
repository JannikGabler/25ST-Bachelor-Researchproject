from pipeline_entities.pipeline.component_entities.component_meta_info.dataclasses.component_meta_info import (
    ComponentMetaInfo,
)
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.attribute_required_constraint import (
    AttributeRequiredConstraint,
)
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.max_predecessors_constraint import (
    MaxPredecessorsConstraint,
)
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.min_predecessors_constraint import (
    MinPredecessorsConstraint,
)

newton_interpolation_core_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"interpolant"},
    attributes_allowed_to_be_overridden={
        "interpolation_nodes",
        "interpolation_values",
        "data_type",
        "node_count",
    },
    pre_dynamic_constraints=[],
    post_dynamic_constraints=[],
    static_constraints=[
        AttributeRequiredConstraint("interpolation_nodes"),
        AttributeRequiredConstraint("interpolation_values"),
        AttributeRequiredConstraint("data_type"),
        AttributeRequiredConstraint("node_count"),
        MinPredecessorsConstraint(1),
        MaxPredecessorsConstraint(1),
    ],
    allow_multiple_executions_for_time_measurements=True,
)
