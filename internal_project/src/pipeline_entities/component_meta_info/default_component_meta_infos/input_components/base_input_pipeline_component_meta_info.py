from pipeline_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo


base_input_pipeline_component_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"data_type", "node_count", "interpolation_interval"},

    dynamic_constraints=[],

    static_constraints=[],

    #mixed_constraints=[]
)