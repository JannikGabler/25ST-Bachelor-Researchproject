import math
import time

import jax
import jax.numpy as jnp



def _internal_perform_action_(nodes) -> jnp.ndarray:
    # Create a square matrix where each entry [j, k] is the difference between node j and node k
    # Note that the diagonal entries are all zero, since each node is subtracted from itself
    pairwise_diff = nodes[:, None] - nodes[None, :]

    # Create a boolean matrix with False on the diagonal and True elsewhere
    # This is used to exclude self-differences (which are zero) from the product
    bool_diff = ~jnp.eye(len(nodes), dtype=bool)

    # Replace diagonal entries (which are zero) with 1.0 to avoid affecting the product
    # Then compute the product across each row (over all k ≠ j)
    product = jnp.prod(jnp.where(bool_diff, pairwise_diff, 1.0), axis=1)

    # Divide 1.0 by the product to get the barycentric weights (Equation (5.6))
    return 1.0 / product


def barycentric_weights_fori(nodes: jnp.ndarray) -> jnp.ndarray:
    n = nodes.shape[0]
    # Ziel-Array initialisieren
    w0 = jnp.empty_like(nodes)

    def body(j, w):
        # 1D-Array aller Differenzen x_j - x_k
        diffs = nodes[j] - nodes
        # Setze die eigene Differenz auf 1, damit sie das Produkt nicht beeinflusst
        diffs = diffs.at[j].set(1.0)
        # Produkt bilden und invertieren
        wj = 1.0 / jnp.prod(diffs)
        return w.at[j].set(wj)

    return jax.lax.fori_loop(0, n, body, w0)



def barycentric_weights_vmap(nodes: jnp.ndarray) -> jnp.ndarray:
    n = nodes.shape[0]
    idx = jnp.arange(n)

    def weight_fn(j, xj):
        diffs = xj - nodes
        diffs = diffs.at[j].set(1.0)
        return 1.0 / jnp.prod(diffs)

    # vmap über (Index, Knotenwert) paart
    return jax.vmap(weight_fn, in_axes=(0, 0))(idx, nodes)




operations = [barycentric_weights_fori, _internal_perform_action_]


for e in range(5):
    node_count: int = 10**e
    slicing_amount = math.ceil(10 / node_count)

    print(f"\n\n--- Node count {node_count} ---")
    nodes: jnp.ndarray = jnp.linspace(-1, 1, node_count, dtype=jnp.float32)
    #values: jnp.ndarray = jnp.cos(nodes)

    print("Compiling...")
    compiled = []
    dummy_array = jnp.empty(node_count, dtype=jnp.float32)

    for op in operations:
        comp = jax.jit(op).lower(dummy_array).compile()
        compiled.append(comp)

    print("Compiled.\n")


    for i, comp in enumerate(compiled):
        durations = []

        for _ in range(50):
            result = comp(nodes)
            result.block_until_ready()

        for _ in range(100):
            start = time.perf_counter()

            result = comp(nodes)
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




