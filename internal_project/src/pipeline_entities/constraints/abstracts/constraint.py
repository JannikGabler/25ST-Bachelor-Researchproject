from abc import ABC, abstractmethod

from pipeline_entities.constraints.enums.constraint_type import ConstraintType


class Constraint(ABC):
    __constraint_type__: ConstraintType



    # def __init__(self):
    #     self.__init_constraint_type__()



    # @abstractmethod
    # def __init_constraint_type__(self):
    #     pass