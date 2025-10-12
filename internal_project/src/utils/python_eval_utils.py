from exceptions.evaluation_error import EvaluationError
from exceptions.not_instantiable_error import NotInstantiableError


class PythonEvalUtils:

    ###################
    ### Constructor ###
    ###################
    def __init__(self) -> None:
        raise NotInstantiableError(
            f"The class {repr(self.__class__.__name__)} cannot be instantiated."
        )

    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def try_expression_evaluation(
        expression: str, name_space: dict[str, object]
    ) -> object:
        try:
            # Security node: __builtins__ are available by default! Might be a security issue (-> define custom safe build ins).
            return eval(expression, {}, name_space)
        except Exception as e:
            raise EvaluationError(f"Error while evaluating {repr(expression)}.")
