import jax.numpy as jnp

from functions.abstracts.compilable_function import CompilableFunction


# TODO
class FastFourierTransformationInterpolant(CompilableFunction):
    """
    TODO
    """

    ###############################
    ### Attributes of instances ###
    ###############################
    # TODO

    ###################
    ### Constructor ###
    ###################
    # TODO

    ##########################
    ### Overridden methods ###
    ##########################
    def _get_internal_evaluate_function_(self, **kwargs) -> callable:
        pass  # TODO

    def __repr__(self) -> str:
        pass  # TODO

    def __str__(self) -> str:
        pass  # TODO

    def __hash__(self) -> int:
        pass  # TODO

    def __eq__(self, other):
        pass  # TODO

    #######################
    ### Private methods ###
    #######################
