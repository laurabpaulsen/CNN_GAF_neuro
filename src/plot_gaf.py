import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



def ax_plot_gaf(ax, gaf):
    ax.imshow(gaf, origin = "lower", cmap = "rainbow")
    ax.set_xticks([])
    ax.set_yticks([])


def main_plotter(gaf_path: Path, save_path:Path = None):
    # load gaf
    gaf = np.load(gaf_path)
    fig, axes = plt.subplots(2, 19, figsize = (20, 3))

    for i in range(19):
        ax_plot_gaf(axes[0, i], gaf[:, :, i, 0])
        ax_plot_gaf(axes[1, i], gaf[:, :, i, 1])

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)

if __name__ in "__main__":
    path = Path(__file__).parents[1]

    gaf_path = path / "data" / "gaf" / "sub-001_9_A.npy"
    plot_path = path / "data" / "gaf_sub-001_A.png"
    main_plotter(gaf_path, save_path=plot_path)