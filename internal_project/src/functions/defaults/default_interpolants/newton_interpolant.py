import jax
import jax.numpy as jnp

from exceptions.invalid_argument_exception import InvalidArgumentException
from functions.abstracts.compilable_function import CompilableFunction


class NewtonInterpolant(CompilableFunction):
    """
    TODO
    """


    ###############################
    ### Attributes of instances ###
    ###############################
    _divided_differences_: jnp.ndarray
    _nodes_: jnp.ndarray


    ###################
    ### Constructor ###
    ###################
    def __init__(self, name: str, nodes: jnp.ndarray, divided_differences: jnp.ndarray):
        super().__init__(name)

        if nodes.shape != divided_differences.shape:
            raise InvalidArgumentException(
                "The shapes of the given nodes and divided_differences arrays differ, although"
                f"they're required to be equal (shape of nodes: {nodes.shape}, shape of divided_differences:"
                f"{divided_differences.shape})."
            )

        self._nodes_ = nodes
        self._divided_differences_ = divided_differences


    ##########################
    ### Overridden methods ###
    ##########################
    def _get_internal_evaluate_function_(self, **kwargs) -> callable:
        return self._internal_evaluate_


    def __repr__(self) -> str:
        return f"NewtonInterpolant(divided_differences={repr(self._divided_differences_)}, nodes={repr(self._nodes_)})"


    def __str__(self) -> str:
        return self.__repr__()


    def __hash__(self) -> int:
        return hash((self._divided_differences_, self._nodes_))


    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        else:
            return (jnp.array_equal(self._divided_differences_, other._divided_differences_, equal_nan=True).item()
                    and jnp.array_equal(self._nodes_, other._nodes_, equal_nan=True).item())


    #######################
    ### Private methods ###
    #######################
    def _internal_evaluate_(self, evaluation_points: jnp.ndarray) -> jnp.ndarray:
        n = self._divided_differences_.size
        nodes: jnp.ndarray = self._nodes_.astype(self._data_type_)
        divided_differences: jnp.ndarray = self._divided_differences_.astype(self._data_type_)

        initial_accumulator: jnp.ndarray = jnp.zeros_like(evaluation_points)

        def horner_step(i, val):
            reverse_index = n - 1 - i
            return val * (evaluation_points - nodes[reverse_index]) + divided_differences[reverse_index]

        return jax.lax.fori_loop(0, n, horner_step, initial_accumulator)
