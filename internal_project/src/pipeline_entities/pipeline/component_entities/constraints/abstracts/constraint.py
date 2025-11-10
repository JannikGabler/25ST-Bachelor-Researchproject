from abc import ABC, abstractmethod

from pipeline_entities.pipeline.component_entities.constraints.enums.constraint_type import ConstraintType


class Constraint(ABC):
    """
    Abstract base class for all pipeline constraints.
    Each constraint belongs to one of the defined constraint types (pre-dynamic, post-dynamic, or static) and provides an error message if its evaluation fails.
    """

    __constraint_type__: ConstraintType


    @abstractmethod
    def get_error_message(self) -> str | None:
        """
        Retrieve the last error message generated during evaluation, if any.

        Returns:
            str | None: Error message if the constraint failed, otherwise None.
        """

        pass
