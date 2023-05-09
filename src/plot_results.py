import matplotlib.pyplot as plt
import numpy as np

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


def get_accuracy(clf_report):
    # get the accuracy from the classification report string
    return float(clf_report.split('\n')[-3].split()[2])

def get_all_accuracies(clf_reports):
    # get the accuracy from the classification report string
    return [get_accuracy(clf_report) for clf_report in clf_reports]

def load_clf_reports(mdl_path):
    clf_reports = []
    subjects = []
    for sub_dir in mdl_path.iterdir():
        clf_report = mdl_path / sub_dir / 'classification_report.txt'
        clf_reports.append(clf_report)
        subjects.append(sub_dir.name)

    # load the classification reports
    clf_reports = [open(clf_report, 'r').read() for clf_report in clf_reports]

    return clf_reports, subjects

def plot_accuracies(accuracies, subjects, save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # sort by subject and turn to percent
    accuracies = [acc*100 for _, acc in sorted(zip(subjects, accuracies))]
    subjects = sorted(subjects)
    
    ax.bar(subjects, accuracies, color = "#82AC85")
    ax.set_ylabel('Accuracy (%)')

    # y limits
    ax.set_ylim([min(accuracies) - 3, max(accuracies)+3])

    # plot mean accuracy
    ax.axhline(np.mean(accuracies), color='k', linestyle='--', label='Mean Accuracy')
    ax.axhline(50, color='k', label='Chance Level')
    ax.legend()

    # turn x labels 45 degrees
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)


def main():
    path = Path(__file__)
    mdl_path = path.parents[1] / 'mdl'
    fig_path = path.parents[1] / 'fig'

    # load the classification reports
    clf_reports, subjects = load_clf_reports(mdl_path)

    # get the accuracies
    accuracies = get_all_accuracies(clf_reports)

    # plot the accuracies
    plot_accuracies(accuracies, subjects, save_path=fig_path / 'accuracies.png')

if __name__ == '__main__':
    main()
