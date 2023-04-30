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
    fig, axes = plt.subplots(3, 15, figsize = (20, 3))

    for i in range(15):
        tmp_gaf = gaf[:, :, i, :]
        ax_plot_gaf(axes[0, i], tmp_gaf[:, :, 0])
        ax_plot_gaf(axes[1, i], tmp_gaf[:, :, 1])
        ax_plot_gaf(axes[2, i], tmp_gaf[:, :, 2])

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)

if __name__ in "__main__":
    path = Path(__file__).parents[1]

    gaf_path = path / "data" / "gaf" / "sub-01_0_0.npy"
    plot_path = path / "data" / "gaf_sub-01_0_0.png"
    main_plotter(gaf_path, save_path=plot_path)