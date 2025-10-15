from enum import Enum


class ConstraintType(Enum):
    """
    Enumeration of constraint types used in the pipeline.

    Attributes:
        PRE_DYNAMIC (int): Constraint applied before dynamic evaluation.
        POST_DYNAMIC (int): Constraint applied after dynamic evaluation.
        STATIC (int): Constraint that remains fixed and does not depend on dynamics.
    """

    PRE_DYNAMIC = 1
    POST_DYNAMIC = 2
    STATIC = 3
