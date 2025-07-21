import math
import time

import jax
import jax.numpy as jnp


def aitken_neville_full_array(nodes: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
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
        return jax.lax.fori_loop(k, n, inner_loop, polynomials_outer)

    # Apply the outer loop to compute all orders of divided differences
    return jax.lax.fori_loop(1, n, outer_loop, initial_polynomials)[n - 1]


def aitken_neville_coeffs(xs: jnp.ndarray, ys: jnp.ndarray) -> jnp.ndarray:
    r"""
    Compute the coefficients of the interpolation polynomial of degree n-1
    passing through points (xs[i], ys[i]) for i=0..n-1 using the Aitken-Neville scheme.

    Returns a 1D array c of length n such that P(x) = c[0] + c[1] x + ... + c[n-1] x^{n-1}.
    """
    n = xs.shape[0]
    # Allocate storage for intermediate polynomials: shape (n, n, n)
    # poly[i, j] holds coefficients for P_{i,j}(x)
    poly = jnp.zeros((n, n, n), dtype=xs.dtype)

    # Initialize P_{i,i}(x) = ys[i]
    poly = poly.at[jnp.arange(n), jnp.arange(n), 0].set(ys)

    # Helper: multiply polynomial by (x - a)
    def mul_x_minus_a(coefs, a):
        # coefs: shape (n,), returns shape (n,)
        # x * p(x): shift right, drop overflow
        shifted = jnp.concatenate([jnp.zeros((1,), dtype=coefs.dtype), coefs[:-1]])
        return shifted - a * coefs

    # Recurrence
    for d in range(1, n):
        for i in range(n - d):
            j = i + d
            # P_{i+1,j} and P_{i,j-1}
            p1 = poly[i+1, j]
            p2 = poly[i, j-1]
            xi = xs[i]
            xj = xs[j]
            # Numerators: (x - xi)*p1 - (x - xj)*p2
            num = mul_x_minus_a(p1, xi) - mul_x_minus_a(p2, xj)
            # Denominator
            denom = xj - xi
            poly = poly.at[i, j].set(num / denom)

    # The full interpolant is P_{0,n-1}
    return poly[0, n-1]





operations = [aitken_neville_full_array, aitken_neville_coeffs]


for e in range(5):
    node_count: int = 10**e
    slicing_amount = math.ceil(10 / node_count)

    print(f"\n\n--- Node count {node_count} ---")
    nodes: jnp.ndarray = jnp.linspace(-1, 1, node_count, dtype=jnp.float32)
    values: jnp.ndarray = jnp.cos(nodes)

    print("Compiling...")
    compiled = []
    dummy_array = jnp.empty(node_count, dtype=jnp.float32)

    for op in operations:
        comp = jax.jit(op).lower(dummy_array, dummy_array).compile()
        compiled.append(comp)

    print("Compiled.\n")


    for i, comp in enumerate(compiled):
        durations = []

        for _ in range(5):
            result = comp(nodes, values)
            result.block_until_ready()

        for _ in range(20):
            start = time.perf_counter()

            result = comp(nodes, values)
            result.block_until_ready()

            end = time.perf_counter()
            durations.append(end - start)

        print(result)

        print(f"Duration {i}: {sum(durations) / len(durations) * 1E03:0.3f} ms")


# print("\n\n\n")
#
# node_count: int = 1000000
# thread_count = 10
# amount_of_nodes_per_thread = math.ceil(node_count / thread_count)
# nodes_per_thread: jnp.ndarray = jnp.empty((thread_count, amount_of_nodes_per_thread,), dtype=jnp.float32)
# nodes = jnp.linspace(-1, 1, node_count, dtype=jnp.float32)
#
# for i in range(thread_count):
#     start_index = i * amount_of_nodes_per_thread
#     end_index = min(node_count, (i + 1) * amount_of_nodes_per_thread)
#     slice: jnp.ndarray = nodes[start_index:end_index]
#     nodes_per_thread = nodes_per_thread.at[i].set(slice)
#
# print("Compiling...")
# dummy_array = jnp.empty((amount_of_nodes_per_thread, ), dtype=jnp.float32)
# compiled_callable = jax.jit(barycentric_weights_fori).lower(dummy_array).compile()
# print("Compiled.\n")
#
# print("Calculation...")
# results: jnp.ndarray = jnp.empty((thread_count, amount_of_nodes_per_thread,), dtype=jnp.float32)
# for i in range(thread_count):
#     print("Starting thread ", i, " ...")
#     results = results.at[i].set(compiled_callable(nodes_per_thread[i]))
#
#
# results.block_until_ready()
#
# print("Calculated.\n")
#
# print(results)




