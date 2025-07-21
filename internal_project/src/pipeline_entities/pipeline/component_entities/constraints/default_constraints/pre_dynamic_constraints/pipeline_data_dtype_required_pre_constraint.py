from dataclasses import fields

import jax.numpy as jnp

from interpolants.abstracts.compilable_interpolant import CompilableInterpolant
from pipeline_entities.pipeline.component_entities.constraints.abstracts.pre_dynamic_constraint import PreDynamicConstraint
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData


class PipelineDataDtypeRequiredPreConstraint(PreDynamicConstraint):
    ###############################
    ### Attributes of instances ###
    ###############################
    _attribute_name_: str



    ###################
    ### Constructor ###
    ###################
    def __init__(self, attribute_name: str) -> None:
        self._attribute_name_ = attribute_name



    ######################
    ### Public methods ###
    ######################
    def evaluate(self, pipeline_data: list[PipelineData], additional_execution_data: AdditionalComponentExecutionData) -> bool:
        for data in pipeline_data:
            if any(field.name == self._attribute_name_ for field in fields(PipelineData)):
                value: object = getattr(data, self._attribute_name_)

                if not isinstance(value, jnp.ndarray) or value.dtype != data.data_type:
                    return False
            else:
                if (not self._attribute_name_ in data.additional_values
                    or not isinstance(data.additional_values[self._attribute_name_], jnp.ndarray)
                    or data.additional_values[self._attribute_name_].dtype != data.data_type):

                    return False

        return True



    def get_error_message(self) -> str | None:
        return "TODO" # TODO



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



