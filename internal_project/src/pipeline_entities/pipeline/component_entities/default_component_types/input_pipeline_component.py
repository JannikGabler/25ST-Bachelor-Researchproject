from abc import ABC

from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component import (
    PipelineComponent,
)


class InputPipelineComponent(PipelineComponent, ABC):
    pass
