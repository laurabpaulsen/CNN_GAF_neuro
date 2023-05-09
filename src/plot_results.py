import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path


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
        subjects.append(sub_dir)

    # load the classification reports
    clf_reports = [open(clf_report, 'r').read() for clf_report in clf_reports]

    return clf_reports, subjects

def main():
    path = Path(__file__)

    mdl_paths = path.parents[1] / 'mdl'

    # load the classification reports
    clf_reports, subjects = load_clf_reports(mdl_paths)

    # get the accuracies
    accuracies = get_all_accuracies(clf_reports)

if __name__ == '__main__':
    main()
