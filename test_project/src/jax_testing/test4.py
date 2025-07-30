import time

import jax
import jax.numpy as jnp


@jax.jit
def hasan(nodes, values) -> jnp.ndarray:
    # Number of interpolation points
    n = nodes.size

    # Initialize coefficients array with function values
    coefficients = values.copy()

    # Compute divided differences of increasing order
    def outer_loop(j, coef_outer):
        # Update entries for current order of divided differences
        def inner_loop(i, coef_inner):
            numerator = coef_outer[i] - coef_outer[i - 1]
            denominator = nodes[i] - nodes[i - j]
            return coef_inner.at[i].set(numerator/denominator)

        # Apply the inner loop to update coefficients for the current order
        coef_outer = jax.lax.fori_loop(j, n, inner_loop, coef_outer)
        return coef_outer

    # Apply the outer loop to compute all orders of divided differences
    coefficients = jax.lax.fori_loop(1, n, outer_loop, coefficients)

    return coefficients



def newton_coeffs(nodes, values) -> jnp.ndarray:
    n = nodes.shape[0]

    # 1) Matrix der paarweisen Differenzen D[i,k] = nodes[i] - nodes[k]
    D = nodes[:, None] - nodes[None, :]       # Shape (n,n)
    #print(D)

    # 2) Baue D' so, dass cumprod über Spalte j exakt prod_{k<j}(x_i - x_k) liefert:
    #    D'[:,0] = 1,  D'[:,j] = D[:, j-1]  für j=1..n-1
    ones = jnp.ones((n, 1), dtype=nodes.dtype)
    Dprime = jnp.concatenate([ones, D[:, :n-1]], axis=1)  # Shape (n,n)
    #print(Dprime)

    # 3) N = kumuliertes Produkt über die Spalten von D'
    N = jnp.cumprod(Dprime, axis=1)  # Shape (n,n), lower triangular
    #print(N)

    # 4) Löse das untere Dreieckssystem N a = values
    #    (kann auch mit jax.scipy.linalg.solve_triangular gemacht werden)
    coeffs = jnp.linalg.solve(N, values)
    return coeffs


@jax.jit
def divided_differences(nodes, values) -> jnp.ndarray:
    n = nodes.shape[0]
    # Wir arbeiten "in-place" auf coeffs
    coeffs = values.copy()

    # Innere Schleife: i von 0 ... n-2
    def body_i(i, carry):
        k, coeffs = carry
        index = n - 1 - i + k
        # Berechnung nur, wenn i < n-k
        update = (coeffs[index] - coeffs[index - 1]) / (nodes[index] - nodes[index - k])
        # select entscheidet, ob wir überschreiben oder nicht
        coeffs = coeffs.at[index].set(update)
        # coeffs = jax.lax.select(
        #     i < (n - k),
        #     coeffs.at[i].set(update),
        #     coeffs
        # )
        return (k, coeffs)

    # Äußere Schleife: k von 1 ... n-1
    def body_k(k, coeffs):
        # Führe inneren Loop über i aus
        _, new_coeffs = jax.lax.fori_loop(
            k,  # start
            n,  # stop (exklusiv)
            body_i,
            (k, coeffs)  # carry: das aktuelle k und coeffs
        )
        return new_coeffs

    # Starte bei k=1 bis k=n-1
    coeffs = jax.lax.fori_loop(1, n, body_k, coeffs)
    return coeffs



def python_loops(nodes, values) -> jnp.ndarray:
    n = nodes.shape[0]
    # Wir arbeiten "in-place" auf coeffs
    coeffs = values.copy()


    for k in range(1, n):
        for i in range(n, k, -1):
            update = (coeffs[i] - coeffs[i - 1]) / (nodes[i] - nodes[i - k])
            coeffs = coeffs.at[i].set(update)


    return coeffs


def jannik(nodes, values) -> jnp.ndarray:
    n = nodes.shape[0]
    # Wir arbeiten "in-place" auf coeffs

    initial_coeffs = values.copy()

    def body_k(k, coeffs):
        new_coeffs = coeffs.copy()

        def body_i(i):
            global new_coeffs

            update = (coeffs[i] - coeffs[i - 1]) / (nodes[i] - nodes[i - k])
            new_coeffs = new_coeffs.at[i].set(update)

        vmapped = jax.vmap(body_i)
        vmapped(jnp.arange(k, n))

        return new_coeffs


    return jax.lax.fori_loop(1, n, body_k, initial_coeffs)


#def divided_differences_eager() -> jnp.ndarray:

# newton_coeffs()
# divided_differences()

# Testaufruf ohne JAX-Tracer:
#result = divided_differences_eager()



#result = jax.jit(divided_differences)()

operations = [hasan, divided_differences, jannik]


for e in range(6):
    node_count: int = 10**e
    print(f"\n\n--- Node count {node_count} ---")
    nodes: jnp.ndarray = jnp.linspace(-1, 1, node_count, dtype=jnp.float32)
    values: jnp.ndarray = jnp.cos(nodes)

    print("Compiling...")
    compiled = []
    dummy_array = jnp.empty(node_count, dtype=jnp.float32)

    for op in operations:
        comp = jax.jit(op).lower(dummy_array, dummy_array).compile()
        compiled.append(comp)

    print("Compiled.\n")


    for i, comp in enumerate(compiled):
        start = time.perf_counter()

        result1 = comp(nodes, values)
        result1.block_until_ready()

        end = time.perf_counter()
        duration = end - start
        print(f"Duration {i}: {duration}")




    # start: float = time.perf_counter()
    # result3 = compiled3(nodes, values)
    # result3.block_until_ready()
    # end: float = time.perf_counter()
    #
    # duration3: float = end - start
    #
    # print(f"Duration of compiled3: {duration3}")

#print(f"Result3: {result3}")

#equal: bool = jax.jit(lambda: jnp.allclose(result1, result2, atol=1E-03, rtol=1E-03))().lower().compile().item() and jax.jit(lambda: jnp.allclose(result2, result3, atol=1E-03, rtol=1E-03))().lower().compile().item()
#equal: bool = jnp.allclose(result1, result2, atol=1E-03, rtol=1E-03).item() and jnp.allclose(result2, result3, atol=1E-03, rtol=1E-03).item()

#print(f"Equal? {equal}")



# equal: bool = jnp.array_equal(result1, result2).item()
# print(equal)


