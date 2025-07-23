from dataclasses import fields

import jax.numpy as jnp

from pipeline_entities.pipeline.component_entities.constraints.abstracts.post_dynamic_constraint import \
    PostDynamicConstraint
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData


class PipelineDataDtypeRequiredPostConstraint(PostDynamicConstraint):
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
    def evaluate(self, input_data: tuple[list[PipelineData], AdditionalComponentExecutionData],
                 output_data: PipelineData) -> bool:

        if any(field.name == self._attribute_name_ for field in fields(PipelineData)):
            value: object = getattr(output_data, self._attribute_name_)

            result = isinstance(value, jnp.ndarray) and value.dtype == output_data.data_type
        else:
            result = (self._attribute_name_ in output_data.additional_values
                    and isinstance(output_data.additional_values[self._attribute_name_], jnp.ndarray)
                    and output_data.additional_values[self._attribute_name_].dtype == output_data.data_type)

        self._error_message_ = None if result else f"Required attribute '{self._attribute_name_}' must be an array with matching data type."
        return result



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



