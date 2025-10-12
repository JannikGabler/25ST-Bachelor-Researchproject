from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import (
    AdditionalComponentExecutionData,
)

if TYPE_CHECKING:
    from data_classes.pipeline_data.pipeline_data import PipelineData

from pipeline_entities.pipeline.component_entities.constraints.abstracts.constraint import (
    Constraint,
)
from pipeline_entities.pipeline.component_entities.constraints.enums.constraint_type import (
    ConstraintType,
)


class PostDynamicConstraint(Constraint, ABC):
    ##########################
    ### Attributs of class ###
    ##########################
    __constraint_type__: ConstraintType = ConstraintType.POST_DYNAMIC

    ######################
    ### Public methods ###
    ######################
    @abstractmethod
    def evaluate(
        self,
        input_data: tuple[list[PipelineData], AdditionalComponentExecutionData],
        output_data: PipelineData,
    ) -> bool:
        pass
