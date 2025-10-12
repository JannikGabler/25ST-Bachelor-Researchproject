from dataclasses import fields

import jax.numpy as jnp

from pipeline_entities.pipeline.component_entities.constraints.abstracts.post_dynamic_constraint import (
    PostDynamicConstraint,
)
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import (
    AdditionalComponentExecutionData,
)
from data_classes.pipeline_data.pipeline_data import PipelineData


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
    def evaluate(
        self,
        input_data: tuple[list[PipelineData], AdditionalComponentExecutionData],
        output_data: PipelineData,
    ) -> bool:

        if any(field.name == self._attribute_name_ for field in fields(PipelineData)):
            value = getattr(output_data, self._attribute_name_)

            if not isinstance(value, jnp.ndarray):
                self._error_message_ = (
                    f"Attribute '{self._attribute_name_}' exists but is not a jax.numpy array. "
                    f"Got type: {type(value)}."
                )
                return False

            if value.dtype != output_data.data_type:
                self._error_message_ = (
                    f"Attribute '{self._attribute_name_}' is a jax.numpy array but has dtype {value.dtype}, "
                    f"expected {output_data.data_type}."
                )
                return False

        else:

            if self._attribute_name_ not in output_data.additional_values:
                self._error_message_ = f"Attribute '{self._attribute_name_}' is missing in additional_values."
                return False

            if not isinstance(
                output_data.additional_values[self._attribute_name_], jnp.ndarray
            ):
                self._error_message_ = (
                    f"Additional value '{self._attribute_name_}' is not a jax.numpy array. "
                    f"Got type: {type(output_data.additional_values[self._attribute_name_])}."
                )
                return False

            if (
                output_data.additional_values[self._attribute_name_].dtype
                != output_data.data_type
            ):
                self._error_message_ = (
                    f"Additional value '{self._attribute_name_}' is a jax.numpy array but has dtype {output_data.additional_values[self._attribute_name_].dtype}, "
                    f"expected {output_data.data_type}."
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
        return isinstance(other, self.__class__)  # Covers None

    # TODO
    # def __hash__(self):
    #     pass
