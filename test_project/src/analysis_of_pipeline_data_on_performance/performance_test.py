import random
import time

import jax.numpy as jnp

from analysis_of_pipeline_data_on_performance.generator_1 import Generator1
from analysis_of_pipeline_data_on_performance.generator_2 import Generator2
from analysis_of_pipeline_data_on_performance.generator_3 import Generator3
from analysis_of_pipeline_data_on_performance.pipeline_data import PipelineData

print("Starting node_count generation...")
node_counts = [random.randint(10, 9999999) for _ in range(0, 150)]
print("Starting interval generation...")
intervals = [(random.uniform(-10, 0), random.uniform(1, 10)) for _ in range(0, 150)]
data_type = jnp.float32


def benchmark_1(warmup=5, runs=50):
    # Warm-up
    for i in range(warmup):
        print(f"{i+1} ", end="")
        generator: Generator1 = Generator1(node_counts[i], data_type, intervals[i])

        generator.generate_nodes().block_until_ready()  # important!

    print("")

    times = []
    for i in range(runs):
        print(f"{i + 1} ", end="")

        generator: Generator1 = Generator1(node_counts[0], data_type, intervals[0])

        start = time.perf_counter()
        generator.generate_nodes().block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    print("\n")

    avg_time = sum(times) / runs
    print(f"Generator 1: {avg_time:.3f} s in average per call (runs = {runs})")
    return avg_time


def benchmark_2(warmup=5, runs=50):
    # Warm-up
    for i in range(warmup):
        print(f"{i+1} ", end="")

        data: PipelineData = PipelineData(node_counts[i], intervals[i], data_type)
        generator: Generator2 = Generator2(data)

        generator.generate_nodes().block_until_ready()  # important!

    print("\n")

    times = []
    for i in range(runs):
        print(f"{i+1} ", end="")

        data: PipelineData = PipelineData(node_counts[0], intervals[0], data_type)
        generator: Generator2 = Generator2(data)

        start = time.perf_counter()
        generator.generate_nodes().block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    print("\n")

    avg_time = sum(times) / runs
    print(f"Generator 2: {avg_time:.3f} s in average per call (runs = {runs})")
    return avg_time


def benchmark_3(warmup=5, runs=50):
    # Warm-up
    for i in range(warmup):
        print(f"{i+1} ", end="")

        data: PipelineData = PipelineData(node_counts[i], intervals[i], data_type)
        generator: Generator3 = Generator3(data)

        generator.generate_nodes().block_until_ready()  # important!

    print("\n")

    times = []
    for i in range(runs):
        print(f"{i+1} ", end="")

        data: PipelineData = PipelineData(node_counts[0], intervals[0], data_type)
        generator: Generator3 = Generator3(data)

        start = time.perf_counter()
        generator.generate_nodes().block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    print("\n")

    avg_time = sum(times) / runs
    print(f"Generator 3: {avg_time:.3f} s in average per call (runs = {runs})")
    return avg_time


r = 50
print("Starting benchmark 1...")
benchmark_1(runs=r)
print("Starting benchmark 2...")
benchmark_2(runs=r)
print("Starting benchmark 3...")
benchmark_3(runs=r)
