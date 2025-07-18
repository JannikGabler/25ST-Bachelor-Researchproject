import jax.numpy as jnp

from interpolants.abstracts.interpolant import Interpolant

# TODO
class FastFourierTransformationInterpolant(Interpolant):
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
    def _get_internal_evaluate_function_(self) -> callable:
        pass # TODO

    def _is_data_type_overriding_required_(self) -> bool:
        pass # TODO

    def _get_data_type_for_no_overriding_(self) -> jnp.dtype:
        pass # TODO



    def __repr__(self) -> str:
        pass # TODO


    def __str__(self) -> str:
        pass # TODO



    def __eq__(self, other):
        pass # TODO



    #######################
    ### Private methods ###
    #######################
