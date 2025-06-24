import time

import jax
import sympy
import jax.numpy as jnp
from sympy import Expr

data_type = jnp.float32
x_array = jnp.linspace(-1.0, 1.0, 101, dtype=data_type)
warmups=10
runs=1000


def get_hard_coded_callable():
    def _func_(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.cos(x) * jnp.exp(x)

    dummy_x = jnp.empty(len(x_array), dtype=data_type)
    return jax.jit(_func_).lower(dummy_x).compile()


def get_sympy_callable(function_string: str, simplify_expression=True) -> callable:
    # 1. Symbolisch parsen mit SymPy
    x = sympy.symbols("x")
    expr: Expr = sympy.sympify(function_string, evaluate=simplify_expression)

    # 2. Lambdify mit JAX-Backend
    f_scalar = sympy.lambdify(x, expr, modules="jax")

    # 3. Vektorisieren (elementweise auf Arrays anwendbar)
    f_vectorized = jax.vmap(f_scalar)

    # 4. Dummy-Eingabe für Kompilierung vorbereiten
    dummy_x = jnp.empty(len(x_array), dtype=data_type)

    # 5. Ahead-of-time kompilieren
    return jax.jit(f_vectorized).lower(dummy_x).compile()


for _ in range(warmups):
    get_hard_coded_callable()(x_array).block_until_ready()
    get_sympy_callable("cos(x) * (E**x)")(x_array).block_until_ready()


start = time.perf_counter()
hard_coded_callable = get_hard_coded_callable()
end = time.perf_counter()
print(f"Compilation of the hard coded callable took: {(end - start) * 1E06:0.1f} µs")

start = time.perf_counter()
sympy_callable = get_sympy_callable("cos(x) * E**x")
end = time.perf_counter()
print(f"Compilation of the sympy parsed callable took: {(end - start) * 1E06:0.1f} µs")


for _ in range(warmups):
    hard_coded_callable(x_array).block_until_ready()
    sympy_callable(x_array).block_until_ready()


times=[]
for _ in range(runs):
    start = time.perf_counter()
    hard_coded_callable(x_array).block_until_ready()
    end = time.perf_counter()
    times.append(end - start)

avg = sum(times) / len(times)
print(f"The hard coded callable took in average: {avg * 1E06:0.1f} µs")


times=[]
for _ in range(runs):
    start = time.perf_counter()
    sympy_callable(x_array).block_until_ready()
    end = time.perf_counter()
    times.append(end - start)

avg = sum(times) / len(times)
print(f"The sympy parsed callable took in average: {avg * 1E06:0.1f} µs")