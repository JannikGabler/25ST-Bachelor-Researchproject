from abc import ABC

from pipeline_entities.components.abstracts.pipeline_component import PipelineComponent


class InputPipelineComponent(PipelineComponent, ABC):
    pass

    # _pipeline_input_: PipelineInput
    #
    #
    #
    # def __init__(self, pipeline_input: PipelineInput, pipeline_data: PipelineData):
    #     super().__init__(pipeline_data)
    #     self._pipeline_input_ = pipeline_input


