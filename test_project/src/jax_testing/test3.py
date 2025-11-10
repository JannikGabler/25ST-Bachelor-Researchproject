import jax
import jax.numpy as jnp

nodes = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]])

print(jnp.pad(nodes, ((1, 0), (0, 0))))

nodes = nodes.transpose()
print(nodes / jnp.array([1, 2]))
