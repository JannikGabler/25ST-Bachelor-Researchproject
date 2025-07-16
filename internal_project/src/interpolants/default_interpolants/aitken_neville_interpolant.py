import jax
import jax.numpy as jnp

from interpolants.abstracts.interpolant import Interpolant


class AitkenNevilleInterpolant(Interpolant):
    _values_: jnp.ndarray
    _nodes_: jnp.ndarray



    def __init__(self, nodes: jnp.ndarray, values: jnp.ndarray):
        self._nodes_ = nodes
        self._values_ = values



    def aitken_neville_scalar(self, nodes: jnp.ndarray, values: jnp.ndarray, x: float) -> jnp.ndarray:
        """
        Computes interpolating polynomial at x

        Args:
            nodes: x coordinates of the given points
            values: y coordinates of the given points
            x: the point at which to evaluate the polynomial

        Returns:
            Interpolated value P(x)
        """
        m = nodes.shape[0]
        # (upper) triangular matrix
        P = jnp.zeros((m, m), dtype=values.dtype)
        # diagonal entries
        P = P.at[jnp.diag_indices(m)].set(values)

        # avoid loops
        # outer loop over interval lengths k = 1, ..., m-1
        def body_k(k, P):
            # inner loop over starting indices i = 0, ..., m-k-1
            def body_i(i, P):
                j = i + k
                # mathematical formula for Aitken-Neville interpolation
                numerator = (x - nodes[i]) * P[i + 1, j] - (x - nodes[j]) * P[i, j - 1]
                denominator = nodes[j] - nodes[i]
                return P.at[i, j].set(numerator / denominator)

            upper = jnp.int32(m - k)
            zero = jnp.int32(0)
            return jax.lax.fori_loop(zero, upper, body_i, P)

        one = jnp.int32(1)
        P = jax.lax.fori_loop(one, m, body_k, P)
        # top right triangle is the result
        return P[0, m - 1]



    def evaluate(self, x:jnp.ndarray) -> jnp.ndarray:
        return jax.jit(jax.vmap(lambda x_i: self.aitken_neville_scalar(self._nodes_, self._values_, x_i)))(x)



    def __repr__(self) -> str:
        return f"AitkenNevilleInterpolant(weights={self._values_}, nodes={self._nodes_})"



    def __str__(self) -> str:
        return self.__repr__()



    def __eq__(self, other):
        if not isinstance(other, AitkenNevilleInterpolant):
            return False
        else:
            return jnp.array_equal(self._nodes_, other._nodes_)
