import inspect
from abc import ABCMeta
from typing import get_type_hints

from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData
from data_classes.pipeline_data.pipeline_data import PipelineData


class PipelineComponentMeta(ABCMeta):
    """
    Metaclass that validates PipelineComponent subclasses.
    Ensures the constructor signature is (pipeline_data: list[PipelineData], additional_execution_info: AdditionalComponentExecutionData).
    """

    def __init__(cls, name, bases, namespace, **kwargs):
        """
        Validate the subclass constructor of a non-abstract PipelineComponent.

        Args:
            name (str): Class name.
            bases (tuple[type, ...]): Base classes.
            namespace (dict): Class namespace.

        Raises:
            TypeError: If the subclass has a constructor with invalid parameters.
        """

        super().__init__(name, bases, namespace)

        if not inspect.isabstract(cls):
            init_method = cls.__init__
            signature = inspect.signature(init_method)
            parameters = list(signature.parameters.values())[1:]

            expected_parameter_types = [list[PipelineData], AdditionalComponentExecutionData]

            hints = get_type_hints(init_method)
            actual_parameter_types = [hints.get(parameter.name, None) for parameter in parameters]

            if len(parameters) != len(expected_parameter_types) or any(actual_type != expected_type for actual_type, expected_type in zip(actual_parameter_types, expected_parameter_types)):
                raise TypeError(f"The subclass '{cls.__name__}' of PipelineComponent has a constructor with invalid parameters. "
                                f"Expected types for the parameters are '{str(expected_parameter_types)}'.")
