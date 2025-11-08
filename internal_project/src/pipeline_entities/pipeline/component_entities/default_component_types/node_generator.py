from abc import ABC
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component import PipelineComponent


class NodeGenerator(PipelineComponent, ABC):
    """
    Abstract base class for node generator components in the pipeline.
    Node generators are responsible for producing interpolation nodes that will be stored in and used by the pipeline data.
    """

    pass
