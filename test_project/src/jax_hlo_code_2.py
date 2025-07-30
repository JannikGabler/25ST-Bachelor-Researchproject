import jax
import jax.numpy as jnp

def f(x, y):
    return x + y - x + 5

inp_x = jnp.ones((1,), dtype=jnp.int32)
inp_y = jnp.zeros((1, ), dtype=jnp.int32)



lowered = jax.jit(f).lower(x, y).compile()

#print(lowered.as_text())

