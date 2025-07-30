import jax
import jax.numpy as jnp


nodes = jnp.array([-1, -0.5, 0, 0.5, 1])
n = len(nodes)

for k in range(1, len(nodes)):
    array1 = nodes[k:]
    array2 = nodes[:n-k]
    print(array1)
    print(array2)
    node_differences: jnp.ndarray = array1 - array2
    print(node_differences)
    print("\n")


# def inner_loop(i: int, polynomials) -> jnp.ndarray:
#     new_polynomials[i] = polynomials[k - i]
#     difference = polynomials[i] - polynomials[i - 1]
#     new_polynomials[i] -= difference * nodes[i] / (nodes[i] - nodes[i - k])
#     new_polynomials[i] += jnp.pad((difference / (nodes[i] - nodes[i - k])), (1, 0))