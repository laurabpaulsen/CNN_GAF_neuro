"""
    Converts the timeseries data into Gramian Angular Fields (both GAFD and GAFS), as well as Markov transition fields. For each timeseries a 3D array containing these are made. 
    The GAFs are saved as numpy arrays in the data/gaf folder.
"""

import numpy as np
from pyts.image import GramianAngularField, MarkovTransitionField
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import multiprocessing as mp

def trial_to_gaf(X:np.ndarray, image_size = 50):
    trans_s = GramianAngularField(method = 'summation', image_size=image_size)
    trans_d = GramianAngularField(method = 'difference', image_size=image_size)
    trans_m = MarkovTransitionField( image_size=image_size)
    # transform each trial into a GAF
    X_gaf_s = trans_s.fit_transform(X)
    X_gaf_d = trans_d.fit_transform(X)
    X_mtf = trans_m.fit_transform(X)


    # loop over gafs per channel 
    for i in range(X_gaf_s.shape[0]):
        gaf = np.stack([X_gaf_s[i], X_gaf_d[i], X_mtf[i]], axis=-1)
        gaf = np.reshape(gaf, (image_size, image_size, 1, 3))

        if i == 0:
            im = gaf

        else:
            im = np.concatenate((im, gaf), axis = 2)

    return im

def gaf_subject(subject:str):
    """
    Converts the timeseries data into Gramian Angular Fields (GAFs) and maps them onto a image with 3 channels.

    Parameters
    ----------
    subject : str
        Subject ID.

    Returns
    -------
    im : np.ndarray
        GAFs of all trials of a subject.
    """
    path = Path(__file__).parents[1]
    npy_path  = path / 'data' / 'preprocessed' / subject 

    # loading in timeseries and labels
    X = np.load(npy_path / "X.npy")
    y = np.load(npy_path / "y.npy")

    # loop over the first 1000 trials per subject
    for i in range(1000):
        gaf = trial_to_gaf(X[i])

        # save each trial as a separate numpy array
        tmp_path = path / "data" / "gaf" / f'{subject}_{i}_{y[i]}.npy'
        np.save(tmp_path, gaf)


def main():
    path = Path(__file__).parents[1]
    preprc_path = path / 'data' /'preprocessed'
    outpath = path / 'data' / 'gaf'

    # loop over subjects
    subjects = [x.name for x in preprc_path.iterdir()]

    # check that outpath exists
    # create directory
    if not outpath.exists():
        outpath.mkdir()

    # make and save GAF/MTFs
    pool = mp.Pool(mp.cpu_count()-1)  # use multiprocessing to speed up the process
    pool.map(gaf_subject, subjects)

if __name__ == '__main__':
    main()