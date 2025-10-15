from functions.abstracts.compilable_function import CompilableFunction


class CallableWrappingFunction(CompilableFunction):
    """
    Wrapper around a user-provided callable. This class allows arbitrary Python callables to be embedded into the pipeline.
    Through the compilation mechanism of the base class, the callable can be JIT-compiled with JAX for efficient evaluation,
    provided that the given callable is JAX-compatible.
    """


    ###############################
    ### Attributes of instances ###
    ###############################
    _callable_: callable


    ###################
    ### Constructor ###
    ###################
    def __init__(self, name: str, cal: callable):
        """
        Args:
            name: Display name of the function.
            cal: A Python callable that maps jax.numpy arrays to jax.numpy arrays.
        """

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
