import jax
import sympy
from sympy import Expr

from functions.abstracts.compilable_function import CompilableFunction


class SingleSympyExpressionFunction(CompilableFunction):
    ###############################
    ### Attributes of instances ###
    ###############################
    _function_expression_: str
    _simplify_expression_: bool

    ###################
    ### Constructor ###
    ###################
    def __init__(self, name: str, function_expression: str, simplify_expression: bool):
        super().__init__(name)

        self._function_expression_ = function_expression
        self._simplify_expression_ = simplify_expression

    ##########################
    ### Overridden methods ###
    ##########################
    def _get_internal_evaluate_function_(self, **kwargs) -> callable:
        return self._create_jax_lambda_()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(function_expression={repr(self._function_expression_)}, "
            f"simplify_expression={repr(self._simplify_expression_)})"
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(function_expression={str(self._function_expression_)}, "
            f"simplify_expression={str(self._simplify_expression_)})"
        )

    def __hash__(self):
        return hash((self._function_expression_, self._simplify_expression_))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        else:
            return (
                self._function_expression_ == other._function_expression_
                and self._simplify_expression_ == other._simplify_expression_
            )

    #######################
    ### Private methods ###
    #######################
    def _create_jax_lambda_(self) -> callable:
        x = sympy.symbols("x")
        expr: Expr = sympy.sympify(
            self._function_expression_, evaluate=self._simplify_expression_
        )  # Although PyCharm shows an error, this is correct

        scalar_lambda = sympy.lambdify(x, expr, modules="jax")

        # Vectorize scalar lambda
        return jax.vmap(scalar_lambda)
