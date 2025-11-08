from abc import ABC
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component import PipelineComponent


class EvaluatorComponent(PipelineComponent, ABC):
    """
    Abstract base class for evaluator components in the pipeline.
    Evaluator components are responsible for computing and attaching evaluation results to the pipeline data.
    """

    pass
