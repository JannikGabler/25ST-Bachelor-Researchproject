from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import AdditionalComponentExecutionData

if TYPE_CHECKING:
    from data_classes.pipeline_data.pipeline_data import PipelineData

from pipeline_entities.pipeline.component_entities.constraints.abstracts.constraint import Constraint
from pipeline_entities.pipeline.component_entities.constraints.enums.constraint_type import ConstraintType



class PostDynamicConstraint(Constraint, ABC):
    """
    Abstract base class for post-dynamic constraints.
    Post-dynamic constraints are evaluated after a component has executed, using both the input data and the resulting output data to verify correctness.
    """


    ##########################
    ### Attributs of class ###
    ##########################
    __constraint_type__: ConstraintType = ConstraintType.POST_DYNAMIC


    ######################
    ### Public methods ###
    ######################
    @abstractmethod
    def evaluate(self, input_data: tuple[list[PipelineData], AdditionalComponentExecutionData], output_data: PipelineData) -> bool:
        """
        Evaluate the constraint after the component's execution.

        Args:
            input_data (tuple[list[PipelineData], AdditionalComponentExecutionData]):
                The input pipeline data and additional execution context provided to the component.
            output_data (PipelineData):
                The pipeline data produced as output by the component.

        Returns:
            bool: True if the constraint is satisfied, False otherwise.
        """

        pass
