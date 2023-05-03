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
    """
    Transform a set of time series into a single image tensor using Gramian Angular Field (GAF) and 
    Markov Transition Field (MTF) techniques.

    Parameters:
    -----------
    X: np.ndarray
        The input time series

    image_size: int
        The size of the output image.

    Returns:
    --------
    np.ndarray
        The transformed image tensor
    """
    trans_s = GramianAngularField(method = 'summation', image_size=image_size)
    trans_d = GramianAngularField(method = 'difference', image_size=image_size)
    trans_m = MarkovTransitionField(image_size=image_size)
    
    # transform each trial
    X_gaf_s = trans_s.fit_transform(X)
    X_gaf_d = trans_d.fit_transform(X)
    X_mtf = trans_m.fit_transform(X)

    # loop over GAFs and MTF per channel
    im = np.stack([X_gaf_s[0], X_gaf_d[0], X_mtf[0]], axis=-1)[:, :, np.newaxis, :]

    for gaf_s, gaf_d, mtf in zip(X_gaf_s[1:], X_gaf_d[1:], X_mtf[1:]):
        gaf = np.stack([gaf_s, gaf_d, mtf], axis=-1)[:, :, np.newaxis, :]
        im = np.concatenate((im, gaf), axis=2)

    return im

def gaf_subject(subject:str, truncate = None):
    """
    Converts the timeseries data into Gramian Angular Fields (GAFs) and maps them onto a image with 3 channels.

    Parameters
    ----------
    subject : str
        Subject ID.
    
    truncate : int or None
        Number of trials to include per subject. If None all trials are included. 

    Returns
    -------
    None
    """
    path = Path(__file__).parents[1]
    npy_path  = path / 'data' / 'preprocessed' / subject 
    outpath = path / 'data' / 'gaf' / subject 

    if not npy_path.exists():
        raise FileNotFoundError(f"Could not find input files for subject {subject}.")
    if not outpath.exists():
        outpath.mkdir()
        

    # loading in timeseries and labels
    X = np.load(npy_path / "X.npy")
    y = np.load(npy_path / "y.npy")


    # loop over the first 1000 trials per subject
    if truncate:
        X = X[:truncate]

    for i, x in enumerate(X):
        gaf = trial_to_gaf(x)

        # save each trial as a separate numpy array
        tmp_path = outpath / f'trial_{i}_label_{y[i]}.npy'
        np.save(tmp_path, gaf)


def main():
    path = Path(__file__).parents[1]
    preprc_path = path / 'data' /'preprocessed'
    outpath = path / 'data' / 'gaf'

    # loop over subjects
    subjects = [x.name for x in preprc_path.iterdir()]

    # check that outpath exists
    if not outpath.exists():
        outpath.mkdir()

    # make and save GAF/MTFs
    pool = mp.Pool(mp.cpu_count()-1)  # use multiprocessing to speed up the process
    pool.map(gaf_subject, subjects)

if __name__ == '__main__':
    main()