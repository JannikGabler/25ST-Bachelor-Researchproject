from abc import ABC, abstractmethod

from pipeline_entities.components.abstracts.pipeline_component import PipelineComponent


class NodeGenerator(PipelineComponent, ABC):
    pass


