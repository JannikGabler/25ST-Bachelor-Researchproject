import jax.numpy as jnp

array1 = jnp.array([1, 2, 3])
array2 = jnp.array([4, 5, 6])

result = array1[:, None] - array2[None, :]

# print(result)
# print(result[0, :])
# print(result[:, 0])
# print(jnp.any(result == -1, axis=1))
# print(result)
# print(array1 * result)
print(array1[None, :])
