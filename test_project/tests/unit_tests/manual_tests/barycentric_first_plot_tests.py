import jax.numpy as jnp
import matplotlib.pyplot as plt

from test_project.src.barycentric_first import barycentric_type1_interpolate


def plot_sin_example():
    interpolation_nodes = jnp.array([0.0, 0.5, 1.0, 1.5])
    function_values = jnp.sin(interpolation_nodes)
    evaluation_points = jnp.linspace(0, 2.0, 200)
    interpolated_values = barycentric_type1_interpolate(evaluation_points, interpolation_nodes, function_values)

    plt.plot(evaluation_points, interpolated_values, label="Baryzentrische Interpolation 1. Art")
    plt.plot(evaluation_points, jnp.sin(evaluation_points), '--', label="Originalfunktion sin(x)")
    plt.scatter(interpolation_nodes, function_values, color='red', label="Stützpunkte")
    plt.title("f(x) = sin(x) auf [0, 2]")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_quadratic_example():
    def f(x): return 2 * x**2 - 3 * x + 1
    interpolation_nodes = jnp.linspace(-5, 10, 15)
    function_values = f(interpolation_nodes)
    evaluation_points = jnp.linspace(-5, 10, 300)
    interpolated_values = barycentric_type1_interpolate(evaluation_points, interpolation_nodes, function_values)

    plt.plot(evaluation_points, interpolated_values, label="Baryzentrische Interpolation 1. Art", color='blue')
    plt.plot(evaluation_points, f(evaluation_points), '--', label="Originalfunktion f(x)", color='green')
    plt.scatter(interpolation_nodes, function_values, color='red', label="Stützpunkte")
    plt.title("f(x) = 2x² - 3x + 1 auf [-5, 10]")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_exp_example():
    def f(x): return jnp.exp(x)
    interpolation_nodes = jnp.linspace(0, 2, 6)
    function_values = f(interpolation_nodes)
    evaluation_points = jnp.linspace(0, 3, 300)
    interpolated_values = barycentric_type1_interpolate(evaluation_points, interpolation_nodes, function_values)

    plt.plot(evaluation_points, interpolated_values, label="Baryzentrische Interpolation 1. Art", color='blue')
    plt.plot(evaluation_points, f(evaluation_points), '--', label="Originalfunktion exp(x)", color='green')
    plt.scatter(interpolation_nodes, function_values, color='red', label="Stützpunkte")
    plt.title("f(x) = exp(x) auf [0, 2]")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_log_example():
    def f(x): return jnp.log(x + 1)
    interpolation_nodes = jnp.linspace(0, 5, 6)
    function_values = f(interpolation_nodes)
    evaluation_points = jnp.linspace(0, 6, 300)
    interpolated_values = barycentric_type1_interpolate(evaluation_points, interpolation_nodes, function_values)

    plt.plot(evaluation_points, interpolated_values, label="Baryzentrische Interpolation 1. Art", color='blue')
    plt.plot(evaluation_points, f(evaluation_points), '--', label="Originalfunktion log(x+1)", color='green')
    plt.scatter(interpolation_nodes, function_values, color='red', label="Stützpunkte")
    plt.title("f(x) = log(x+1) auf [0, 5]")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_sin_example()
    plot_quadratic_example()
    plot_exp_example()
    plot_log_example()


