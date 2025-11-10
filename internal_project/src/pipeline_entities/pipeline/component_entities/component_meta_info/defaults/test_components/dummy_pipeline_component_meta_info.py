from pipeline_entities.pipeline.component_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo


"""
Dummy component meta information used for testing or placeholder purposes.
This meta info does not modify any attributes, allows no overrides, and defines no constraints. Multiple executions 
for time measurements are allowed.
"""
dummy_pipeline_component_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying=set(),

    attributes_allowed_to_be_overridden=set(),

    pre_dynamic_constraints=[],

    post_dynamic_constraints=[],

    static_constraints=[],

    allow_multiple_executions_for_time_measurements=True)
