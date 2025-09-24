"""Visualisation of Majorana-string Encodings."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def symplectic_matshow(symplectics, title: str | None = None):
    """Colorised Matplolibn.matshow of a symplectic array."""
    # Crear el mapa de colores cualitativo
    colors = ["linen", "tab:red", "tab:blue", "tab:purple"]

    left, right = np.hsplit(symplectics, 2)
    symplectics = (left + 2 * right) % 4

    # For cases where there is no Y or Z, the colormap must be scaled down.
    cmap = ListedColormap(colors[: np.max(symplectics) + 1])
    plt.matshow(symplectics, cmap=cmap, alpha=0.8)
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
    labels = ["I", "X", "Z", "Y"]
    plt.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4)
    if title is not None:
        plt.title(title)
    plt.xlabel("Qubit")
    plt.tick_params(labeltop=False, labelbottom=True)
    plt.ylabel(r"Majorana $\gamma$")
    plt.yticks(range(symplectics.shape[0]))
