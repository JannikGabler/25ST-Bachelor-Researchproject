from pipeline_entities.component_meta_info.default_component_meta_infos.test_components.dummy_pipeline_component_meta_info import \
    dummy_pipeline_component_meta_info
from pipeline_entities.components.abstracts.node_generator import NodeGenerator

from pipeline_entities.components.abstracts.pipeline_component import PipelineComponent
from pipeline_entities.components.decorators.pipeline_component import pipeline_component


@pipeline_component(id="dummy", type=PipelineComponent, meta_info=dummy_pipeline_component_meta_info)
class DummyPipelineComponent(NodeGenerator):
    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineComponent:
        pass



    ##########################
    ### Overridden methods ###
    ##########################
    def __repr__(self) -> str:
        return "Dummy pipeline component with no functionality"