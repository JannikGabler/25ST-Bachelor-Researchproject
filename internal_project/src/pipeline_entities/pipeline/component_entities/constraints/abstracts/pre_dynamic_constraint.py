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


class PreDynamicConstraint(Constraint, ABC):
    ##########################
    ### Attributs of class ###
    ##########################
    __constraint_type__: ConstraintType = ConstraintType.PRE_DYNAMIC

    ######################
    ### Public methods ###
    ######################
    @abstractmethod
    def evaluate(
        self,
        pipeline_data: list[PipelineData],
        additional_execution_data: AdditionalComponentExecutionData,
    ) -> bool:
        pass
