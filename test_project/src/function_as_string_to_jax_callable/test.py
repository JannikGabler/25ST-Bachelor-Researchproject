import jax
import sympy
import jax.numpy as jnp
from sympy import Expr


def create_callable(function_string: str, simplify_expression=True) -> callable:
    # 1. Symbolisch parsen mit SymPy
    x = sympy.symbols("x")
    expr: Expr = sympy.sympify(function_string, evaluate=simplify_expression)

    # 2. Lambdify mit JAX-Backend
    f_scalar = sympy.lambdify(x, expr, modules="jax")

    # 3. Vektorisieren (elementweise auf Arrays anwendbar)
    f_vectorized = jax.vmap(f_scalar)

    # 4. Dummy-Eingabe für Kompilierung vorbereiten
    dummy_input = jnp.linspace(0.0, 1.0, 10)

    # 5. Ahead-of-time kompilieren
    return jax.jit(f_vectorized).lower(dummy_input).compile()


# function_string: str = input("-> ")
function_string: str = "cos(x) * (2 ** x)"


jax_callable = create_callable(function_string)
print(jax_callable)
