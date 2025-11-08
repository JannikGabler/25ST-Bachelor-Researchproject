import time

import jax
import sympy
import jax.numpy as jnp
from sympy import Expr

data_type = jnp.float32
x_array = jnp.linspace(0, 3, 101, dtype=data_type)
warmups = 10
runs = 10000


# def get_hard_coded_callable():
#     def _func_(x: jnp.ndarray) -> jnp.ndarray:
#         return jnp.cos(x)
#
#     dummy_x = jnp.empty(len(x_array), dtype=data_type)
#     return jax.jit(_func_).lower(dummy_x).compile()


def get_hard_coded_callable_piecewise():
    def _func_(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(x <= 1, jnp.sin(x), jnp.where(x <= 2, jnp.exp(x), jnp.cos(x)))

    dummy_x = jnp.empty(len(x_array), dtype=data_type)
    return jax.jit(_func_).lower(dummy_x).compile()


# def get_sympy_callable(function_string: str, simplify_expression=True) -> callable:
#     # 1. Symbolisch parsen mit SymPy
#     x = sympy.symbols("x")
#     expr: Expr = sympy.sympify(function_string, evaluate=simplify_expression)
#
#     # 2. Lambdify mit JAX-Backend
#     f_scalar = sympy.lambdify(x, expr, modules="jax")
#
#     # 3. Vektorisieren (elementweise auf Arrays anwendbar)
#     f_vectorized = jax.vmap(f_scalar)
#
#     # 4. Dummy-Eingabe für Kompilierung vorbereiten
#     dummy_x = jnp.empty(len(x_array), dtype=data_type)
#
#     # 5. Ahead-of-time kompilieren
#     return jax.jit(f_vectorized).lower(dummy_x).compile()


def get_sympy_callable_piecewise(
    piecewise_function_strings: list[tuple[tuple[any, any], str]],
    simplify_expression=True,
) -> callable:
    # 1. Symbolisch parsen mit SymPy
    x = sympy.symbols("x")
    expressions: list[tuple[Expr, any]] = []

    for (lower, upper), function_string in piecewise_function_strings:
        expression: Expr = sympy.sympify(function_string, evaluate=simplify_expression)
        condition = (x >= lower) & (x < upper)
        expressions.append((expression, condition))

    complete_expression = sympy.Piecewise(*expressions)

    # 2. Lambdify mit JAX-Backend
    f_scalar = sympy.lambdify(x, complete_expression, modules="jax")

    # 3. Vektorisieren (elementweise auf Arrays anwendbar)
    f_vectorized = jax.vmap(f_scalar)

    # 4. Dummy-Eingabe für Kompilierung vorbereiten
    dummy_x = jnp.empty(len(x_array), dtype=data_type)

    # 5. Ahead-of-time kompilieren
    return jax.jit(f_vectorized).lower(dummy_x).compile()


for _ in range(warmups):
    get_hard_coded_callable_piecewise()(x_array).block_until_ready()
    get_sympy_callable_piecewise(
        [((0, 1), "sin(x)"), ((1, 2), "exp(x)"), ((2, 3), "cos(x)")]
    )(x_array).block_until_ready()


start = time.perf_counter()
hard_coded_callable_piecewise = get_hard_coded_callable_piecewise()
end = time.perf_counter()
print(
    f"Compilation of the hard coded piecewise callable took: {(end - start) * 1E06:0.1f} µs"
)

start = time.perf_counter()
sympy_callable_piecewise = get_sympy_callable_piecewise(
    [((0, 1), "sin(x)"), ((1, 2), "exp(x)"), ((2, 3), "cos(x)")]
)
end = time.perf_counter()
print(
    f"Compilation of the sympy parsed piecewise callable took: {(end - start) * 1E06:0.1f} µs"
)


for _ in range(warmups):
    hard_coded_callable_piecewise(x_array).block_until_ready()
    sympy_callable_piecewise(x_array).block_until_ready()


times = []
for _ in range(runs):
    start = time.perf_counter()
    hard_coded_callable_piecewise(x_array).block_until_ready()
    end = time.perf_counter()
    times.append(end - start)

avg = sum(times) / len(times)
print(f"The hard coded piecewise callable took in average: {avg * 1E06:0.1f} µs")
print(f"Cost: {hard_coded_callable_piecewise.cost_analysis()}")

times = []
for _ in range(runs):
    start = time.perf_counter()
    sympy_callable_piecewise(x_array).block_until_ready()
    end = time.perf_counter()
    times.append(end - start)

avg = sum(times) / len(times)
print(f"The sympy parsed piecewise callable took in average: {avg * 1E06:0.1f} µs")
print(f"Cost: {sympy_callable_piecewise.cost_analysis()}")
