import matplotlib.pyplot as plt
import numpy as np

factor = 1

def plot(function) -> None:
    plt.figure(figsize=(10, 6))

    x_list = np.linspace(-1, 1, 100)
    y_list = function(x_list)

    y_min = min(y_list)
    y_max = max(y_list)
    d = y_max - y_min
    l_border = np.floor(y_min - factor * d)
    u_border = np.ceil(y_max + factor * d)

    plt.plot(x_list, y_list)

    plt.yticks(np.linspace(l_border, u_border, 3 + 4 * factor))
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot(lambda x: np.sin(10 * x))
plot(lambda x: 1 / (1 + 25 * x * x))





