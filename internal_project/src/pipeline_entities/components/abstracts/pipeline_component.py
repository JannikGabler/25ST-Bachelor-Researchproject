from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING

from pipeline_entities.components.meta_classes.pipeline_component_meta import PipelineComponentMeta
from pipeline_entities.data_transfer.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.data_transfer.pipeline_data import PipelineData

if TYPE_CHECKING:
    from pipeline_entities.component_info.dataclasses.pipeline_component_info import PipelineComponentInfo


class PipelineComponent(metaclass=PipelineComponentMeta):
    ###########################
    ### Attributes of class ###
    ###########################
    _info_: PipelineComponentInfo



    ###############################
    ### Attributes of instances ###
    ###############################
    _pipeline_data_: list[PipelineData]
    _additional_execution_info_: AdditionalComponentExecutionData



    ###################
    ### Constructor ###
    ###################
    def __init__(self, pipeline_data: list[PipelineData], additional_execution_info: AdditionalComponentExecutionData) -> None:
        self._pipeline_data_ = pipeline_data
        self._additional_execution_info_ = additional_execution_info



    ########################
    ### Abstract methods ###
    ########################
    @abstractmethod
    def perform_action(self) -> PipelineData:
        pass



    #########################
    ### Getters & setters ###
    #########################
    @property
    def info(self) -> PipelineComponentInfo:
        return self._info_ # PipelineComponentInfo is immutable

