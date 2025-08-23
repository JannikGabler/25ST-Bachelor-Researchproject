from dataclasses import fields

import jax.numpy as jnp

from functions.abstracts.compilable_function import CompilableFunction
from pipeline_entities.pipeline.component_entities.constraints.abstracts.pre_dynamic_constraint import PreDynamicConstraint
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData


class PipelineDataDtypeRequiredPreConstraint(PreDynamicConstraint):
    ###############################
    ### Attributes of instances ###
    ###############################
    _attribute_name_: str
    _error_message_: str | None



    ###################
    ### Constructor ###
    ###################
    def __init__(self, attribute_name: str) -> None:
        self._attribute_name_ = attribute_name
        self._error_message_ = None



    ######################
    ### Public methods ###
    ######################
    def evaluate(self, pipeline_data: list[PipelineData], additional_execution_data: AdditionalComponentExecutionData) -> bool:
        for data in pipeline_data:
            if any(field.name == self._attribute_name_ for field in fields(PipelineData)):
                value: object = getattr(data, self._attribute_name_, None)

                if not isinstance(value, jnp.ndarray):
                    self._error_message_ = (
                        f"Attribute '{self._attribute_name_}' in PipelineData must be a jnp.ndarray, "
                        f"but got type '{type(value).__name__}'."
                    )
                    return False

                if value.dtype != data.data_type:
                    self._error_message_ = (
                        f"The dtype of attribute '{self._attribute_name_}' does not match the expected pipeline data type. "
                        f"Expected: {data.data_type}, got: {value.dtype}."
                    )
                    return False

            else:
                if self._attribute_name_ not in data.additional_values:
                    self._error_message_ = (
                        f"Required attribute '{self._attribute_name_}' not found in 'additional_values'."
                    )
                    return False

                if not isinstance(data.additional_values[self._attribute_name_], jnp.ndarray):
                    self._error_message_ = (
                        f"Attribute '{self._attribute_name_}' in 'additional_values' must be a jnp.ndarray, "
                        f"but got type '{type(data.additional_values[self._attribute_name_]).__name__}'."
                    )
                    return False

                if data.additional_values[self._attribute_name_].dtype != data.data_type:
                    self._error_message_ = (
                        f"The dtype of attribute '{self._attribute_name_}' in 'additional_values' does not match the expected pipeline data type. "
                        f"Expected: {data.data_type}, got: {data.additional_values[self._attribute_name_].dtype}."
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
        return f"{self.__class__.__name__}"

    def __str__(self):
        return self.__repr__()



    def __eq__(self, other):
        return isinstance(other, self.__class__) # Covers None



    # TODO
    # def __hash__(self):
    #     pass



