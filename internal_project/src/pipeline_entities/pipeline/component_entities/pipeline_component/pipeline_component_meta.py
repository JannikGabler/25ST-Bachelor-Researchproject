import inspect
from abc import ABCMeta
from typing import get_type_hints

from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from data_classes.pipeline_data.pipeline_data import PipelineData


class PipelineComponentMeta(ABCMeta):

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace)

        if not inspect.isabstract(cls):
            init_method = cls.__init__
            signature = inspect.signature(init_method)
            parameters = list(signature.parameters.values())[1:]   # Skip 'self' parameter

            # Usage of module name and class name to prevent circular import
            expected_parameter_types = [list[PipelineData], AdditionalComponentExecutionData]

            hints = get_type_hints(init_method)
            actual_parameter_types = [hints.get(parameter.name, None) for parameter in parameters]

            if len(parameters) != len(expected_parameter_types) \
                or any(actual_type != expected_type for actual_type, expected_type in zip(actual_parameter_types, expected_parameter_types)):

                raise TypeError(f"The subclass '{cls.__name__}' of PipelineComponent has a constructor with invalid parameters. "
                                f"Expected types for the parameters are '{str(expected_parameter_types)}'.")








