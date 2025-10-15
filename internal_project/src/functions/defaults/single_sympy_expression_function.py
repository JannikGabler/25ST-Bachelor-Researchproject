import jax
import sympy
from sympy import Expr

from functions.abstracts.compilable_function import CompilableFunction


class SingleSympyExpressionFunction(CompilableFunction):
    """
    Compilable function that represents a single mathematical expression. The class uses SymPy to parse a string into a
    symbolic expression, optionally simplifies it, and then converts the result into a JAX-compatible callable.
    """


    ###############################
    ### Attributes of instances ###
    ###############################
    _function_expression_: str
    _simplify_expression_: bool


    ###################
    ### Constructor ###
    ###################
    def __init__(self, name: str, function_expression: str, simplify_expression: bool):
        """
        Args:
            name: Display name of the function.
            function_expression: The mathematical expression as a string.
            simplify_expression: Whether SymPy should simplify expressions on parsing.
        """

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
            return self._function_expression_ == other._function_expression_ and self._simplify_expression_ == other._simplify_expression_


    #######################
    ### Private methods ###
    #######################
    def _create_jax_lambda_(self) -> callable:
        x = sympy.symbols("x")
        expr: Expr = sympy.sympify(self._function_expression_, evaluate=self._simplify_expression_)

        scalar_lambda = sympy.lambdify(x, expr, modules="jax")

        return jax.vmap(scalar_lambda)
