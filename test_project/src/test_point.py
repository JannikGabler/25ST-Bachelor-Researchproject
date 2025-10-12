import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(10, 6))

ax.plot([-1, 0, 1], [0, 1, 0])
ax.plot(0, 1.005, "ro", transform=ax.get_xaxis_transform(), clip_on=False)

fig.show()
