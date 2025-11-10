import time
from functools import partial

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

data_type = jnp.float16


def generate_nodes_1(node_count: int) -> jnp.ndarray:
    return generate_nodes_1_core(
        jnp.arange(1, 2 * node_count - 1, step=2, dtype=data_type)
    )


@jax.jit
def generate_nodes_1_core(nodes):
    nodes = jnp.multiply(nodes, jnp.pi / (2 * len(nodes)))

    return jnp.cos(nodes)


@partial(jax.jit, static_argnames=("node_count",))
def generate_nodes_2(node_count: int) -> jnp.ndarray:
    # Rescale int array for better stability
    nodes: jnp.ndarray = jnp.arange(1, 2 * node_count - 1, step=2, dtype=data_type)
    nodes = jnp.multiply(nodes, jnp.pi / (2 * node_count))

    return jnp.cos(nodes)


@partial(jax.jit, static_argnames=("node_count",))
def generate_nodes_3(node_count: int) -> jnp.ndarray:
    first_half: jnp.ndarray = jnp.arange(1, 2 * node_count - 1, step=2, dtype=data_type)
    first_half = jnp.multiply(first_half, jnp.pi / (2 * node_count))
    first_half = jnp.cos(first_half)

    second_half: jnp.ndarray = -jnp.flip(first_half)

    return jnp.concatenate([first_half, second_half])


@partial(jax.jit, static_argnames=("node_count",))
def generate_nodes_4(node_count: int) -> jnp.ndarray:
    # Rescale int array for better stability
    nodes = jax.lax.iota(data_type, node_count)[::2] + 1
    nodes = jnp.multiply(nodes, jnp.pi / (2 * node_count))

    return jnp.cos(nodes)


@partial(jax.jit, static_argnames=("node_count",))
def generate_nodes_5(node_count: int) -> jnp.ndarray:
    # Schrittweite δ = π / n
    δ = jnp.pi / node_count
    # Erzeuge k = [0,1,...,n-1]
    k = jnp.arange(node_count, dtype=data_type)
    # Winkel = k*δ + δ/2  (entspricht (2k+1)π/(2n))
    angles = k * δ + δ * 0.5
    # Kosinus-Funktion
    return jnp.cos(angles)


def benchmark(function, node_count: int, warmup=5, runs=50):
    # Warm-up
    for _ in range(warmup):
        function(node_count).block_until_ready()  # important!

    times = []
    for i in range(runs):
        start = time.perf_counter()
        function(node_count).block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / runs
    print(
        f"{function.__name__}: {avg_time * 1e6:.2f} µs in average per call (runs = {runs})"
    )
    return avg_time * 1e6


node_counts = [10 ** (i + 1) for i in range(7)]
times_matrix = [[], [], [], [], []]
for i, node_count in enumerate(node_counts):
    times_matrix[0].append(benchmark(generate_nodes_1, node_count, warmup=5, runs=50))
    times_matrix[1].append(benchmark(generate_nodes_2, node_count, warmup=5, runs=50))
    times_matrix[2].append(benchmark(generate_nodes_3, node_count, warmup=5, runs=50))
    times_matrix[3].append(benchmark(generate_nodes_4, node_count, warmup=5, runs=50))
    times_matrix[4].append(benchmark(generate_nodes_5, node_count, warmup=5, runs=50))


for i, times_array in enumerate(times_matrix):
    plt.plot(node_counts, times_array, label=f"{i+1}")

plt.loglog()
plt.legend()
plt.show()
