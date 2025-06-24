import random
import time

import jax.numpy as jnp

from benchmark_aot_vs_jit.aot_generator import AOTGenerator
from benchmark_aot_vs_jit.jit_generator import JitGenerator
from benchmark_aot_vs_jit.pipeline_data import PipelineData
from benchmark_aot_vs_jit.pure_python_generator import PurePythonGenerator

node_count = 1000000
data_type = jnp.float32
interval = jnp.array([-1, 1], dtype=data_type)

data_1: PipelineData = PipelineData(node_count, data_type)
data_2: PipelineData = PipelineData(node_count, data_type)
data_3: PipelineData = PipelineData(node_count, data_type)



# Init generators
start = time.perf_counter()
aot_generator: AOTGenerator = AOTGenerator(data_1)
end = time.perf_counter()
print(f"Init of the AOT generator took: {(end - start) * 1E06:0.1f} µs")

start = time.perf_counter()
jit_generator: JitGenerator = JitGenerator(data_2)
end = time.perf_counter()
print(f"Init of the JIT generator took: {(end - start) * 1E06:0.1f} µs")

start = time.perf_counter()
python_generator: PurePythonGenerator = PurePythonGenerator(data_3)
end = time.perf_counter()
print(f"Init of the PYT generator took: {(end - start) * 1E06:0.1f} µs\n")



# Set interval after init
data_1.interpolation_interval = jnp.array([-1, 1], dtype=data_type)
data_2.interpolation_interval = jnp.array([-1, 1], dtype=data_type)
data_3.interpolation_interval = jnp.array([-1, 1], dtype=data_type)


# Generate nodes
start = time.perf_counter()
aot_generator.generate_nodes()
end = time.perf_counter()
print(f"Node generation took the AOT generator: {(end - start) * 1E03:0.1f} ms")

start = time.perf_counter()
jit_generator.generate_nodes()
end = time.perf_counter()
print(f"Node generation took the JIT generator: {(end - start) * 1E03:0.1f} ms")

start = time.perf_counter()
python_generator.generate_nodes()
end = time.perf_counter()
print(f"Node generation took the PYT generator: {(end - start) * 1E03:0.1f} ms\n")


#Output
# print("Calculated notes by the AOT generator")
# print(str(data_1.nodes) + "\n")
# print("Calculated notes by the JIT generator")
# print(str(data_2.nodes) + "\n")
# print("Calculated notes by the PYT generator")
# print(str(data_3.nodes) + "\n")


















