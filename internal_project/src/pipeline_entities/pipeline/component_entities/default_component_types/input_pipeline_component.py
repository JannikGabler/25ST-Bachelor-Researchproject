from abc import ABC
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component import PipelineComponent


class InputPipelineComponent(PipelineComponent, ABC):
    """
    Abstract base class for input components in the pipeline.
    Input components are responsible for injecting or transferring input data into the pipeline data structure.
    """

    pass
