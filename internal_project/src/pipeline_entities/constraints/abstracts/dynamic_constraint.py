from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pipeline_entities.pipeline_input.pipeline_input import PipelineInput

if TYPE_CHECKING:
    from pipeline_entities.data_transfer.pipeline_data import PipelineData

from pipeline_entities.constraints.abstracts.constraint import Constraint
from pipeline_entities.constraints.enums.constraint_type import ConstraintType


class DynamicConstraint(Constraint, ABC):
    ##########################
    ### Attributs of class ###
    ##########################
    __constraint_type__: ConstraintType = ConstraintType.DYNAMIC



    ######################
    ### Public methods ###
    ######################
    @abstractmethod
    def evaluate(self, pipeline_data: PipelineData, pipeline_input: PipelineInput) -> bool:
        pass


