import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# set parameters for plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['image.cmap'] = 'RdBu_r'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.dpi'] = 300


def ax_plot_gaf(ax, gaf):
    ax.imshow(gaf, origin = "lower", vmin = -1, vmax = 1)
    ax.set_xticks([])
    ax.set_yticks([])


def main_plotter(gaf_path: Path, save_path:Path = None):
    # load gaf
    gaf = np.load(gaf_path)
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(3, 11, width_ratios=[10] * 10 + [0.5])
    axes = gs.subplots(sharex=True, sharey=True)

    for i in range(10):
        tmp_gaf = gaf[:, :, i, :]
        im = ax_plot_gaf(axes[0, i], tmp_gaf[:, :, 0])
        ax_plot_gaf(axes[1, i], tmp_gaf[:, :, 1])
        ax_plot_gaf(axes[2, i], tmp_gaf[:, :, 2])
        axes[2, i].set_xlabel(i+1)

    axes[0,0].set_ylabel("GAFS")
    axes[1,0].set_ylabel("GAFD")
    axes[2,0].set_ylabel("MTF")

    fig.suptitle("Gramian Angular and Markov Transition Fields")
    fig.supxlabel("Sensor")

    # Add colorbar subplot
    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(im, cax=cax)

    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cbar = cax.figure.colorbar(
        mpl.cm.ScalarMappable(norm=norm),
        cax=cax,
        ticks = [-1, 0, 1])


    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

if __name__ in "__main__":
    path = Path(__file__).parents[1]

    gaf_path = path / "data" / "gaf" / "sub-01" / "trial_0_label_0.npy"
    plot_path = path / "fig"

    if not plot_path.exists():
        plot_path.mkdir()

    main_plotter(gaf_path, save_path=plot_path / "gaf_sub-01_0_0.png")