import dill
import jax
import jax.numpy as jnp


f = lambda x: x**2

inp = jnp.arange(10)

jax_compiled_f = jax.jit(f).lower(inp).compile()

data1 = dill.dumps(jax_compiled_f)
data2 = dill.dumps(f)

print(data1)
print(data2)