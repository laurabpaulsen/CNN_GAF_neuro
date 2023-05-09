import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path


def get_accuracy(clf_report):
    # get the accuracy from the classification report string
    return float(clf_report.split('\n')[5].split()[1])

def get_all_accuracies(clf_reports):
    # get the accuracy from the classification report string
    return [get_accuracy(clf_report) for clf_report in clf_reports]

def load_clf_reports(mdl_path):

    # find all the classification reports
    clf_reports = list(mdl_path.glob('**/classification_report.txt'))

    # load the classification reports
    clf_reports = [open(clf_report, 'r').read() for clf_report in clf_reports]

    return clf_reports

def main():
    path = Path(__file__)

    mdl_paths = path.parents[2] / 'mdl'

    # load the classification reports
    clf_reports = load_clf_reports(mdl_paths)

    # get the accuracies
    accuracies = get_all_accuracies(clf_reports)

    # print the accuracies
    print(accuracies)

if __name__ == '__main__':
    main()
