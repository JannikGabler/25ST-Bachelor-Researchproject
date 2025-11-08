from abc import ABC
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component import PipelineComponent


class InterpolationCore(PipelineComponent, ABC):
    """
    Abstract base class for interpolation core components in the pipeline.
    Interpolation cores are responsible for computing the coefficients, constructing interpolants based on provided data and attaching them to the pipeline data.
    """

    pass
