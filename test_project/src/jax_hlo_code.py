import jax
import jax.numpy as jnp


def f(x):
    return x + 5


inp_x = jnp.ones((2,), dtype=jnp.int32)


lowered = jax.jit(f).lower(x).compile()

# print(lowered.as_text())
