from pipeline_entities.pipeline.component_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.pre_dynamic_constraints.input_key_required_constraint import InputKeyRequiredConstraint
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.max_predecessors_constraint import MaxPredecessorsConstraint
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.min_predecessors_constraint import MinPredecessorsConstraint


"""
Component meta information for the function callable input component. 
This component modifies the attribute original_function and allows it to be overridden. 
It has a pre-dynamic constraint that requires the key function_callable to be provided in the pipeline input, while it has no post-dynamic constraints. 
Furthermore, it enforces through static constraints that the component has exactly one predecessor. 
Multiple executions for time measurements are not allowed.
"""
function_callable_input_component_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"original_function"},

    attributes_allowed_to_be_overridden={"original_function"},

    pre_dynamic_constraints=[InputKeyRequiredConstraint("function_callable")],

    post_dynamic_constraints=[],

    static_constraints=[MinPredecessorsConstraint(1), MaxPredecessorsConstraint(1)],

    allow_multiple_executions_for_time_measurements=False)
