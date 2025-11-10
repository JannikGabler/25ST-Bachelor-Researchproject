import time

import jax
import jax.numpy as jnp

nodes = jnp.linspace(-1, 1, 10000)
values = jnp.cos(nodes)


def f1() -> jnp.ndarray:
    indices: jnp.ndarray = jnp.arange(len(nodes), dtype=jnp.int32)

    one = jnp.astype(1.0, jnp.float32)
    minus_one = jnp.astype(-1.0, jnp.float32)
    two = jnp.astype(2.0, jnp.float32)

    weights: jnp.ndarray = jnp.where(indices % 2 == 0, one, minus_one)

    weights = weights.at[0].divide(two)
    weights = weights.at[-1].divide(two)

    return weights


def f2() -> jnp.ndarray:
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


compiled1 = jax.jit(f1).lower().compile()
compiled2 = jax.jit(f2).lower().compile()

durations = []
for _ in range(100):
    start = time.perf_counter()

    result = compiled1()
    result.block_until_ready()

    end = time.perf_counter()
    durations.append(end - start)

print(f"{sum(durations) / len(durations) * 1E03:0.3f} ms")


durations = []
for _ in range(100):
    start = time.perf_counter()

    result = compiled2()
    result.block_until_ready()

    end = time.perf_counter()
    durations.append(end - start)

print(f"{sum(durations) / len(durations) * 1E03:0.3f} ms")
