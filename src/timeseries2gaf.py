"""
    Converts the timeseries data into Gramian Angular Fields (GAFs) and maps them onto a image with 3 channels.
    The GAFs are saved as numpy arrays in the data/gaf folder.
"""

import numpy as np
from pyts.image import GramianAngularField
from pathlib import Path
from tqdm import tqdm

def trial_to_gaf(X, image_size):
    trans_s = GramianAngularField(method = 'summation')
    trans_d = GramianAngularField(method = 'summation')
    # transform each trial into a GAF
    X_gaf_s = trans_s.fit_transform(X)
    X_gaf_d = trans_d.fit_transform(X)

    # loop over gafs per channel 
    for i in range(X_gaf_s.shape[0]):
        gaf = np.concatenate([X_gaf_s[i], X_gaf_d[i], np.zeros((50, 50))])
        gaf = np.reshape(gaf, (50,50,1,3))

        if i == 0:
            im = gaf

        else:
            im = np.concatenate((im, gaf), axis = 2)

    return im

def gaf_subject(subpath, outpath, image_size):

    timeseries = subpath.glob('*timeseries*')

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

    for subject in tqdm(subjects):
        outpath = path.parents[1] / 'data' / 'gaf' / subject.name

        # create directory
        if not outpath.exists():
            outpath.mkdir(parents=True)

        gaf_subject(subject, outpath, image_size)


if __name__ == '__main__':
    main()