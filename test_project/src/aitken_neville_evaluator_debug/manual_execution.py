import jax
import jax.numpy as jnp

from jax.typing import DTypeLike


data_type: DTypeLike = jnp.float32
nodes = jnp.linspace(-1, 1, 4, dtype=data_type)
values = nodes ** 3
evaluation_points = jnp.array([-1, -0.5, 0, 0.5, 1])
n: int = len(nodes)
m: int = len(evaluation_points)

# Initialize values array with function values
initial_values = values[:, None] * jnp.ones((1, m))

def outer_loop(k, polynomials_outer):

    def inner_loop(i, polynomials_inner):
        value_differences = polynomials_outer[i] - polynomials_outer[i - 1]
        quotients = (evaluation_points - nodes[i]) / (nodes[i] - nodes[i - k])
        return polynomials_inner.at[i].set(polynomials_outer[i] + quotients * value_differences)

    # val2 = initial_values
    # for i in range(k, n):
    #     val2 = inner_loop(i, val2)
    # return val2
    return jax.lax.fori_loop(k, n, inner_loop, polynomials_outer)

# val1 = initial_values
# for k in range(1, n):
#     val1 = outer_loop(k, val1)
# result = val1[n - 1]

result = jax.lax.fori_loop(1, n, outer_loop, initial_values)[n - 1]

print(result)