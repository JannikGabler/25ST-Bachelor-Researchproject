import numpy as np
import matplotlib.pyplot as plt

def equidistant_points(n):
    """Equidistante Punkte im Intervall [-1, 1]."""
    return np.linspace(-1, 1, n)

def chebyshev_points_1st_kind(n):
    """Tschebyscheff-Punkte 1. Art."""
    k = np.arange(1, n + 1)
    return np.cos((2*k - 1) * np.pi / (2 * n))

def chebyshev_points_2nd_kind(n):
    """Tschebyscheff-Punkte 2. Art."""
    k = np.arange(0, n)
    return np.cos(k * np.pi / (n - 1))

def plot_support_points(x_points, title, n_points):
    """Zeichnet die obere Kreishälfte und markiert die Stützstellen."""
    # Berechne obere Kreishälfte
    x = np.linspace(-1, 1, 400)
    y = np.sqrt(1 - x**2)

    fig, ax = plt.subplots()

    # Halbkreis
    ax.plot(x, y, color="purple", linewidth=2)

    # Stützstellen auf dem Kreis
    y_points = np.sqrt(1 - x_points**2)

    # Vertikale Linien zu den Punkten
    for i, xi in enumerate(x_points):
        color = "tab:red" if i % 2 == 0 else "tab:blue"
        ax.plot([xi, xi], [0, np.sqrt(1 - xi**2)], linestyle="--", color=color, alpha=0.7)

    # Punkte markieren
    ax.scatter(x_points, y_points, s=50, color="black", zorder=5)
    ax.scatter(x_points, np.zeros_like(x_points), s=50, facecolors="none", edgecolors="black", zorder=5)

    # Achsen & Layout
    ax.axhline(0, color="gray", linewidth=1)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(0, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title(f"{title}\n(n={n_points})", fontsize=13)

    # Optional: Textformel und Achsenbeschriftung
    ax.text(-0.9, 0.9, r"$x^2 + y^2 = 1$", fontsize=14, color="purple")
    ax.text(1.05, 0, "x", fontsize=12)
    ax.text(0, 1.05, "y", fontsize=12)

    plt.tight_layout()
    plt.show()

def main(n_points):
    plot_support_points(equidistant_points(n_points), "Equidistante Punkte", n_points)
    plot_support_points(chebyshev_points_1st_kind(n_points), "Tschebyscheff-Punkte 1. Art", n_points)
    plot_support_points(chebyshev_points_2nd_kind(n_points), "Tschebyscheff-Punkte 2. Art", n_points)

if __name__ == "__main__":
    main(n_points=8)
