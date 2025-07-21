from pipeline_entities.pipeline.component_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo


dummy_pipeline_component_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying=set(),

    attributes_allowed_to_be_overridden=set(),

    pre_dynamic_constraints=[],

    post_dynamic_constraints=[],

    static_constraints=[],
)

