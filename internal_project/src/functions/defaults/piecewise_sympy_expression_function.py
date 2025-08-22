import jax
import sympy
from sympy import Expr

from functions.abstracts.compilable_function import CompilableFunction


class PiecewiseSympyExpressionFunction(CompilableFunction):
    ###############################
    ### Attributes of instances ###
    ###############################
    _function_expressions_: list[tuple[tuple[float, float], str]]
    _simplify_expression_: bool



    ###################
    ### Constructor ###
    ###################
    def __init__(self, name: str, function_expressions: list[tuple[tuple[float, float], str]], simplify_expression: bool) -> None:
        super().__init__(name)

        self._function_expressions_ = function_expressions
        self._simplify_expression_ = simplify_expression



    ##########################
    ### Overridden methods ###
    ##########################
    def _get_internal_evaluate_function_(self, **kwargs) -> callable:
        return self._create_jax_lambda_()



    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(function_expressions={repr(self._function_expressions_)}, "
                f"simplify_expression={repr(self._simplify_expression_)})")

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}(function_expressions={str(self._function_expressions_)}, "
                f"simplify_expression={str(self._simplify_expression_)})")



    def __hash__(self):
        return hash((self._function_expressions_, self._simplify_expression_))



    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        else:
            return self._function_expressions_ == other._function_expressions_ and self._simplify_expression_ == other._simplify_expression_



    #######################
    ### Private methods ###
    #######################
    def _create_jax_lambda_(self) -> callable:
        x = sympy.symbols("x")
        piecewise_sympy_expression: Expr = self._create_piecewise_sympy_expression(x)

        lambda_scalar = sympy.lambdify(x, piecewise_sympy_expression, modules="jax")

        return jax.vmap(lambda_scalar)



    def _create_piecewise_sympy_expression(self, x: object) -> Expr:
        expressions: list[tuple[Expr, any]] = []

        for (lower, upper), function_expression in self._function_expressions_:
            expression: Expr = sympy.sympify(function_expression, evaluate=self._simplify_expression_) # Although PyCharm shows an error, this is correct
            condition = (x >= lower) & (x < upper)
            expressions.append((expression, condition))

        return sympy.Piecewise(*expressions)