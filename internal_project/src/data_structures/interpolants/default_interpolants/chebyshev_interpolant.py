from data_structures.interpolants.abstracts.interpolant import Interpolant
import jax
import jax.numpy as jnp


class ChebyshevInterpolant(Interpolant):
    def __init__(self, nodes: jnp.ndarray, values: jnp.ndarray, weights: jnp.ndarray):
        self.nodes = nodes
        self.values = values
        self.weights = weights

    @jax.jit
    def _interpolate_single(self, x: float) -> jnp.ndarray:
        diffs = x - self.nodes
        bool_diffs = jnp.equal(diffs, 0.0)
        exact_value = jnp.sum(jnp.where(bool_diffs, self.values, 0.0))
        updated_diffs = jnp.where(bool_diffs, 1.0, diffs)
        node_polynomial = jnp.prod(diffs)
        interpolated_value = node_polynomial * jnp.sum((self.weights / updated_diffs) * self.values)
        return jnp.where(jnp.any(bool_diffs), exact_value, interpolated_value)

    def evaluate(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self._interpolate_single)(x)
