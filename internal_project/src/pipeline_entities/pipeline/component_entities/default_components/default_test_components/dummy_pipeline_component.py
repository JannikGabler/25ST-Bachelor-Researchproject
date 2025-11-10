from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.test_components.dummy_pipeline_component_meta_info import dummy_pipeline_component_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.node_generator import NodeGenerator
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component import PipelineComponent
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component


@pipeline_component(id="dummy", type=PipelineComponent, meta_info=dummy_pipeline_component_meta_info)
class DummyPipelineComponent(NodeGenerator):
    """
    Dummy pipeline component for testing and demonstration. Provides no functionality and simply serves as a placeholder in the pipeline.
    """

    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineComponent:
        """
        Perform the component’s action.

        Returns:
            PipelineComponent: No meaningful result, placeholder only.
        """

        pass


    ##########################
    ### Overridden methods ###
    ##########################
    def __repr__(self) -> str:
        """
        String representation of the dummy component.

        Returns:
            str: Description.
        """

        return "Dummy pipeline component with no functionality"
