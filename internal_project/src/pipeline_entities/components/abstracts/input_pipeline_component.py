from pipeline_entities.components.abstracts.pipeline_component import PipelineComponent
from pipeline_entities.components.meta_classes.pipeline_component_meta import PipelineComponentMeta
from pipeline_entities.data_transfer.pipeline_data import PipelineData
from pipeline_entities.pipeline_input.pipeline_input import PipelineInput


class InputPipelineComponent(PipelineComponent, metaclass=PipelineComponentMeta):
    _pipeline_input_: PipelineInput



    def __init__(self, pipeline_input: PipelineInput, pipeline_data: PipelineData):
        super().__init__(pipeline_data)
        self._pipeline_input_ = pipeline_input


