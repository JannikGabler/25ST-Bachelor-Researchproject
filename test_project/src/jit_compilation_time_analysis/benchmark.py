import random
import time
from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp

from jit_compilation_time_analysis.pipeline_data import PipelineData
from utils import jax_utils


@jax.jit
def generate_nodes(data: PipelineData) -> jnp.ndarray:
    # Rescale int array for better stability
    nodes: jnp.ndarray = jnp.arange(1, 2 * data.node_count + 1, step=2, dtype=data_type)
    nodes = jnp.multiply(nodes, jnp.pi / (2 * data.node_count))

    nodes = jnp.cos(nodes)

    jax_utils.rescale_array_to_interval(nodes, (-1, 1), data.interpolation_interval)

    return nodes




node_count = random.randint(100, 9999999)
interval = (random.uniform(-100, 0), random.uniform(0, 100))
data_type = jnp.float32

data_1: PipelineData = PipelineData(node_count, interval, data_type)
data_2 = deepcopy(data_1)


start = time.perf_counter()
generate_nodes(data_1)
end = time.perf_counter()
print(f"Generation time 1: {(end - start) * 1E06:0.1f} µs")

start = time.perf_counter()
generate_nodes(data_2)
end = time.perf_counter()
print(f"Generation time 2: {(end - start) * 1E06:0.1f} µs")

