import jax
import jax.numpy as jnp


def aitken_neville_full_array(nodes: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    n = nodes.size

    # Initialize coefficients array with function values
    initial_polynomials = jnp.zeros((n, n)) + values

    # Compute divided differences of increasing order
    def outer_loop(k, polynomials_outer):
        # Update entries for current order of divided differences
        def inner_loop(i, polynomials_inner):
            node_difference: jnp.ndarray = nodes[i] - nodes[i - k]
            polynomial_difference: jnp.ndarray = polynomials_outer[i] - polynomials_outer[i - 1]

            summand1: jnp.ndarray = polynomials_outer[i]
            summand2: jnp.ndarray = - nodes[i] / node_difference * polynomial_difference
            summand3: jnp.ndarray = jnp.roll(polynomial_difference / node_difference, 1, axis=0)

            return polynomials_inner.at[i].set(summand1 + summand2 + summand3)

        # Apply the inner loop to update coefficients for the current order
        return jax.lax.fori_loop(k, n, inner_loop, polynomials_outer)

    # Apply the outer loop to compute all orders of divided differences
    return jax.lax.fori_loop(1, n, outer_loop, initial_polynomials)[n]


def f(nodes: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    n = nodes.size

    # Initialize coefficients array with function values
    initial_polynomials = jnp.zeros((n, n)).at[:, 0].set(values)

    # Compute divided differences of increasing order
    def outer_loop(k, polynomials_outer):
        # Update entries for current order of divided differences
        def inner_loop(i, polynomials_inner):
            node_difference: jnp.ndarray = nodes[i] - nodes[i - k]
            polynomial_difference: jnp.ndarray = polynomials_outer[i] - polynomials_outer[i - 1]

            summand1: jnp.ndarray = polynomials_outer[i]
            summand2: jnp.ndarray = - nodes[i] / node_difference * polynomial_difference
            summand3: jnp.ndarray = jnp.roll(polynomial_difference / node_difference, 1, axis=0)

            return polynomials_inner.at[i].set(summand1 + summand2 + summand3)

        # Apply the inner loop to update coefficients for the current order
        #return jax.lax.fori_loop(k, n, inner_loop, polynomials_outer)
        result = polynomials_outer
        for i in range(k, n):
            result = inner_loop(i, result)
        return result

    # Apply the outer loop to compute all orders of divided differences
    result = initial_polynomials
    for k in range(1, n):
        result = outer_loop(k, result)
    return result[n - 1]


nodes = jnp.linspace(-1, 1, 4, dtype=jnp.float32)
values = jnp.power(nodes, 3)

print(f(nodes, values))
print(aitken_neville_full_array(nodes, values))