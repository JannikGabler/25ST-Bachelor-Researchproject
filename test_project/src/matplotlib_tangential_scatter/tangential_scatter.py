import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import transforms

fig, ax = plt.subplots()

# Grid-Parameter
origin_x, origin_y = 0, 0
cols, rows = 5, 4
radius_px = 20  # Radius in Bildschirm-Pixeln

for i in range(cols):
    for j in range(rows):
        # Mittelpunkt in Datenkoordinaten
        x = origin_x + i * 2
        y = origin_y + j * 2

        # Kreis in Display-Koordinaten erzeugen (Pixel)
        circle = mpatches.Circle(
            (0, 0),
            radius_px,
            transform=transforms.IdentityTransform(),
            fill=False,
            edgecolor="blue",
        )

        # Kreis an richtige Datenposition verschieben
        circle.set_transform(
            ax.transData + plt.gca().transData.inverted() + ax.transData
        )
        # ^^ Trick: zuerst "keine Skalierung", dann mit Datenkoordinaten kombinieren

        # Verschieben auf die richtige Stelle
        circle.set_center((x, y))

        ax.add_patch(circle)

# Zusätzlich normale Datenplot-Funktion
ax.plot([0, 10], [0, 20], color="red")

plt.show()
