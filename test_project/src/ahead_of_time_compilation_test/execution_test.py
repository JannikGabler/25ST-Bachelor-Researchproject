import random
import jax.numpy as jnp

from ahead_of_time_compilation_test.aot_generator import AOTGenerator
from ahead_of_time_compilation_test.pipeline_data import PipelineData

node_count = 5 #random.randint(10, 1000)
data_type = jnp.float32
interval = jnp.array([-1, 1], dtype=data_type)

data: PipelineData = PipelineData(node_count, data_type)

generator: AOTGenerator = AOTGenerator(data)

data.interpolation_interval = interval

generator.generate_nodes()

print(data.nodes)











