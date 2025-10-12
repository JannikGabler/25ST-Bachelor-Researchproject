from abc import ABC, abstractmethod

from pipeline_entities.pipeline.component_entities.constraints.enums.constraint_type import (
    ConstraintType,
)


class Constraint(ABC):
    __constraint_type__: ConstraintType

    @abstractmethod
    def get_error_message(self) -> str | None:
        pass

    # def __init__(self):
    #     self.__init_constraint_type__()

    # @abstractmethod
    # def __init_constraint_type__(self):
    #     pass
