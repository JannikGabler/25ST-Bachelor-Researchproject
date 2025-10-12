from abc import ABC, abstractmethod

from pipeline_entities.pipeline.component_entities.constraints.enums.constraint_type import (
    ConstraintType,
)


class Constraint(ABC):
    __constraint_type__: ConstraintType

    @abstractmethod
    def get_error_message(self) -> str | None:
        pass
