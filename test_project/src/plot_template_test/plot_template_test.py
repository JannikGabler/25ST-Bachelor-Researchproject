from copy import deepcopy

from data_classes.plot_template.plot_template import PlotTemplate, AxesWrapper, FigureWrapper

#
# template: PlotTemplate = PlotTemplate(figsize=(6, 4))
#
# template.ax.set_title("Hallo")
# template.ax.plot([0, 1, 2], [1, 0, 1], label="Test")
# template.fig.legend(True)
#
# template.fig.show()

template: PlotTemplate = PlotTemplate(figsize=(6,4))   # erzeugt später plt.subplots(figsize=(6,4))
# Alles wird nur aufgezeichnet:

# ax: AxesWrapper = template.ax
# fig: FigureWrapper = template.fig

template.ax.plot([0,1,2,3], [0,1,4,9], label='quad')
template.ax.set_xlabel("x")
template.ax.set_ylabel("y")
template.ax.legend()
template.ax.grid()
template.fig.suptitle("Deferred Example")

# Später: tatsächlich rendern und anzeigen
template.fig.show()


template_copy = deepcopy(template)

template_copy.fig.show()


