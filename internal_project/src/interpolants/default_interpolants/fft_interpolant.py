from data_structures.interpolants.abstracts.interpolant import Interpolant
import jax
import jax.numpy as jnp


class FastFourierTransformationInterpolant(Interpolant):
    _weights_: jnp.ndarray
    _nodes_: jnp.ndarray
    _interval_: jnp.ndarray



    def __init__(self, nodes: jnp.ndarray, weights: jnp.ndarray, interval: jnp.ndarray):
        self._nodes_ = nodes
        self._weights_ = weights
        self._interval_ = interval



    def evaluate(self, x: jnp.ndarray) -> jnp.ndarray:
        pass



    def __repr__(self) -> str:
        return f"FFTInterpolant(weights={self._weights_}, nodes={self._nodes_})"



    def __str__(self) -> str:
        return self.__repr__()



    def __eq__(self, other):
        if not isinstance(other, FastFourierTransformationInterpolant):
            return False
        else:
            return jnp.array_equal(self._weights_, other._weights_) and jnp.array_equal(self._nodes_, other._nodes_)
