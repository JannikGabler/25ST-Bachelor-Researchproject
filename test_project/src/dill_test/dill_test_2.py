import dill
import jax.numpy as jnp

from functions.defaults.callable_wrapping_function import CallableWrappingFunction
from data_classes.pipeline_data.pipeline_data import PipelineData


def func(x):
    return jnp.cos(x) * jnp.exp(x)


pipeline_data: PipelineData = PipelineData()

pipeline_data.node_count = 5
pipeline_data.data_type = jnp.float32
pipeline_data.interpolation_interval = jnp.array([-1, 1], dtype=pipeline_data.data_type)

pipeline_data.original_function = CallableWrappingFunction("Original function", func)

data = dill.dumps(pipeline_data)
print(data)

restored_pipeline_data: PipelineData = dill.loads(data)
print(restored_pipeline_data)


