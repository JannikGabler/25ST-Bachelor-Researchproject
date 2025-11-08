from exceptions.evaluation_error import EvaluationError
from exceptions.not_instantiable_error import NotInstantiableError


class PythonEvalUtils:
    """
    Utility helpers for evaluating Python expressions with a provided namespace. This class is not meant to be instantiated.
    """


    ###################
    ### Constructor ###
    ###################
    def __init__(self) -> None:
        """
        Raises:
            NotInstantiableError: Always raised when initialized to indicate that this class is not meant to be instantiated.
        """

        raise NotInstantiableError(f"The class {repr(self.__class__.__name__)} cannot be instantiated.")


    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def try_expression_evaluation(expression: str, name_space: dict[str, object]) -> object:
        """
        Evaluates an expression.

        Args:
            expression (str): Expression to evaluate.
            name_space (dict[str, object]): Symbols available during evaluation.

        Returns:
            object: Result of the evaluation.

        Raises:
            EvaluationError: If evaluation fails.
        """

        try:
            return eval(expression, {}, name_space)
        except Exception as e:
            raise EvaluationError(f"Error while evaluating {repr(expression)}.")
