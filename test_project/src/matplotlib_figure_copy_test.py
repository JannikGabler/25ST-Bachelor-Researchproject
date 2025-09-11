import pickle
from copy import deepcopy

import dill
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot([1, 2, 3, 4], [10, 20, 25, 30])
ax.set_title("Beispiel 1")

ax.set_xlabel("x")
ax.set_ylabel("y")

fig.show()

# buf = dill.dumps(fig)
# fig_copy = dill.loads(buf)

fig_copy = plt.Figure(fig)

ax.set_title("Beispiel 2")
fig_copy.show()