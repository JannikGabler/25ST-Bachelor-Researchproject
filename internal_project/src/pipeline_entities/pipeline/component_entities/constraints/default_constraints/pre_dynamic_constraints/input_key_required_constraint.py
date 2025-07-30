from dataclasses import fields

from pipeline_entities.pipeline.component_entities.constraints.abstracts.pre_dynamic_constraint import PreDynamicConstraint
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.large_data_classes.pipeline_input.pipeline_input import PipelineInput
from pipeline_entities.large_data_classes.pipeline_input.pipeline_input_data import PipelineInputData


class InputKeyRequiredConstraint(PreDynamicConstraint):
    ###############################
    ### Attributes of instances ###
    ###############################
    _key_: str
    _error_message_: str | None



    ###################
    ### Constructor ###
    ###################
    def __init__(self, key: str) -> None:
        if key.startswith("_") and key.endswith("_"):
            self._key_ = key[1:-1]
        else:
            self._key_ = key
        self._error_message_ = None



    ######################
    ### Public methods ###
    ######################
    def evaluate(self, pipeline_data: list[PipelineData], additional_execution_data: AdditionalComponentExecutionData) -> bool:
        pipeline_input: PipelineInput = additional_execution_data.pipeline_input

        if any(field.name == self._key_ for field in fields(PipelineInputData)):
            transformed_key = f"_{self._key_}_"
            value = getattr(pipeline_input, transformed_key, None)

            if value is None:
                self._error_message_ = (
                    f"Required input field '{transformed_key}' is present but has value None."
                )
                return False

        else:

            if self._key_ in pipeline_input.additional_values:
                if pipeline_input.additional_values[self._key_] is None:
                    self._error_message_ = (
                        f"Key '{self._key_}' found in 'additional_values', but its value is None."
                    )
                    return False

            elif self._key_ in pipeline_input.additional_directly_injected_values:
                if pipeline_input.additional_directly_injected_values[self._key_] is None:
                    self._error_message_ = (
                        f"Key '{self._key_}' found in 'additional_directly_injected_values', but its value is None."
                    )
                    return False

            else:
                self._error_message_ = (
                    f"Required input key '{self._key_}' is missing from both 'additional_values' and 'additional_directly_injected_values'."
                )
                return False

        self._error_message_ = None
        return True


    def get_error_message(self) -> str | None:
        return self._error_message_



    ##########################
    ### Overridden methods ###
    ##########################
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(key={repr(self._key_)})"

    def __str__(self):
        return self.__repr__()



    def __hash__(self):
        return hash(self._key_)



    def __eq__(self, other):
        if not isinstance(other, self.__class__):   # Covers None
            return False
        else:
            return self._key_ == other._key_
