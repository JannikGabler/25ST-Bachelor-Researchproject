import jax
import jax.numpy as jnp

from exceptions.invalid_argument_exception import InvalidArgumentException
from functions.abstracts.compilable_function import CompilableFunction


class BarycentricFirstInterpolant(CompilableFunction):
    """
    TODO
    """


    ###############################
    ### Attributes of instances ###
    ###############################
    _nodes_: jnp.ndarray
    _values_: jnp.ndarray
    _weights_: jnp.ndarray


    ###################
    ### Constructor ###
    ###################
    def __init__(self, name: str, nodes: jnp.ndarray, values: jnp.ndarray, weights: jnp.ndarray):
        super().__init__(name)

        if len({nodes.shape, values.shape, weights.shape}) >= 2:
            raise InvalidArgumentException(
                "The shapes of the given nodes, values and weight arrays differ, although "
                f"they're required to be equal (shape of nodes: {nodes.shape}, shape of values: {values.shape}, shape of "
                f"weights: {weights.shape})."
            )

        self._nodes_ = nodes
        self._values_ = values
        self._weights_ = weights


    ##########################
    ### Overridden methods ###
    ##########################
    def _get_internal_evaluate_function_(self, **kwargs) -> callable:
        return self._internal_evaluate_


    def __repr__(self) -> str:
        return (
            f"BarycentricFirstInterpolant(nodes={repr(self._nodes_)}, values={repr(self._values_)}, "
            f"weights={repr(self._weights_)})"
        )


    def __str__(self) -> str:
        return self.__repr__()


    def __hash__(self) -> int:
        return hash((self._nodes_, self._values_, self._weights_))


    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        else:
            return (jnp.array_equal(self._nodes_, other._nodes_, equal_nan=True).item() and jnp.array_equal(self._values_, other._values_, equal_nan=True).item()
                    and jnp.array_equal(self._weights_, other._weights_, equal_nan=True).item())


    #######################
    ### Private methods ###
    #######################
    def _internal_evaluate_(self, evaluation_points: jnp.ndarray) -> jnp.ndarray:
        nodes: jnp.ndarray = self._nodes_.astype(self._data_type_)
        values: jnp.ndarray = self._values_.astype(self._data_type_)
        weights: jnp.ndarray = self._weights_.astype(self._data_type_)

        def calc_exact_match_value(exact_matches: jnp.ndarray) -> jnp.ndarray:
            exact_match_index: jnp.ndarray = jnp.argmax(exact_matches)
            return values[exact_match_index]

        def calc_node_polynomial_value(point: jnp.ndarray) -> jnp.ndarray:
            return jnp.prod(point - nodes)

        def calc_sum(differences: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum((weights * values) / differences)

        def _evaluate_single_(point: jnp.ndarray) -> jnp.ndarray:
            differences: jnp.ndarray = point - nodes
            exact_matches: jnp.ndarray = differences == 0.0

            return jnp.where(jnp.any(exact_matches), calc_exact_match_value(exact_matches), calc_node_polynomial_value(point) * calc_sum(differences))

        return jax.vmap(_evaluate_single_)(evaluation_points)

