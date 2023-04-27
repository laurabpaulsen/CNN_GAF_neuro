"""
    Converts the timeseries data into Gramian Angular Fields (GAFs) and maps them onto a image with 3 channels.
    The GAFs are saved as numpy arrays in the data/gaf folder.
"""

import numpy as np
from pyts.image import GramianAngularField
from pathlib import Path
from tqdm import tqdm

def trial_to_gaf(X, image_size):
    transformer_sum = GramianAngularField(image_size, method='summation')
    transformer_dif = GramianAngularField(image_size, method='difference')
    # transform each trial into a GAF
    X_gasf = transformer_sum.fit_transform(X)
    X_gadf = transformer_dif.fit_transform(X)
    X_gaf = np.stack((X_gasf, X_gadf, np.zeros(X_gadf.shape)), axis=3)

    return X_gaf

def gaf_subject(sub_path, out_path, image_size):

    timeseries = sub_path.glob('*timeseries*')

    for i, ts in enumerate(timeseries):
        timeseries_path = sub_path / ts

        file_name_label = 'labels' + str(ts).split("timeseries")[-1]
        labels_path = sub_path / file_name_label
        
        # loading in timeseries and labels
        X = np.load(timeseries_path)
        y = np.load(labels_path)

        # loop over all trials
        for j in range(X.shape[0]):

            # for now only include 2000 first per participant
            if j < 2000:
                gaf = trial_to_gaf(X[j], image_size)

                # save each trial as a separate numpy array
                tmp_path = out_path / f'run_{1+i}_trial_{j}_label_{y[j]}.npy'
                np.save(tmp_path, gaf)


def main():
    path = Path(__file__)
    
    image_size = 26

    data_path = path.parents[1] / 'data' / 'preprocessed'

    # loop over subjects
    subjects = [x for x in data_path.iterdir() if x.is_dir()]

    for subject in subjects:
        out_path = path.parents[1] / 'data' / 'gaf' / subject.name

        # create directory
        if not out_path.exists():
            out_path.mkdir(parents=True)

        gaf_subject(subject, out_path, image_size)


if __name__ == '__main__':
    main()