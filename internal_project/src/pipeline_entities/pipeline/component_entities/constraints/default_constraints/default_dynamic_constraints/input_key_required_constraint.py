from dataclasses import fields

from pipeline_entities.pipeline.component_entities.constraints.abstracts.dynamic_constraint import DynamicConstraint
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.large_data_classes.pipeline_input.pipeline_input import PipelineInput
from pipeline_entities.large_data_classes.pipeline_input.pipeline_input_data import PipelineInputData


class InputKeyRequiredConstraint(DynamicConstraint):
    ###############################
    ### Attributes of instances ###
    ###############################
    _key_: str



    ###################
    ### Constructor ###
    ###################
    def __init__(self, key: str) -> None:
        if key.startswith("_") and key.endswith("_"):
            self._key_ = key[1:-1]
        else:
            self._key_ = key



    ##########################
    ### Overridden methods ###
    ##########################
    def evaluate(self, pipeline_data: list[PipelineData], additional_execution_data: AdditionalComponentExecutionData) -> bool:
        pipeline_input: PipelineInput = additional_execution_data.pipeline_input

        if any(field.name == self._key_ for field in fields(PipelineInputData)):
            transformed_key = f"_{self._key_}_"
            return getattr(pipeline_input, transformed_key)
        else:
            return self._key_ in pipeline_input.additional_values or self._key_ in pipeline_input.additional_directly_injected_values



    def __eq__(self, other):
        if not isinstance(other, InputKeyRequiredConstraint):   # Covers None
            return False
        else:
            return self._key_ == other._key_



    def __hash__(self):
        return hash(self._key_)



    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(key='{self._key_}')"