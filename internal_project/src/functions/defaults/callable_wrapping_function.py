import jax
import sympy
from sympy import Expr

from functions.abstracts.compilable_function import CompilableFunction


class CallableWrappingFunction(CompilableFunction):
    ###############################
    ### Attributes of instances ###
    ###############################
    _callable_: callable



    ###################
    ### Constructor ###
    ###################
    def __init__(self, name: str, cal: callable):
        super().__init__(name)

        self._callable_ = cal



    ##########################
    ### Overridden methods ###
    ##########################
    def _get_internal_evaluate_function_(self, **kwargs) -> callable:
        return self._callable_



    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(callable={repr(self._callable_)})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(callable={str(self._callable_)})"



    def __hash__(self):
        return hash(self._callable_)



    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        else:
            return self._callable_ == other._callable_

