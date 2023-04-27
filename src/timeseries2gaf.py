"""
    Converts the timeseries data into Gramian Angular Fields (GAFs) and maps them onto a image with 3 channels.
    The GAFs are saved as numpy arrays in the data/gaf folder.
"""

import numpy as np
from pyts.image import GramianAngularField
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def trial_to_gaf(X:np.ndarray):
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

def gaf_subject(subject:str, outpath:Path, label:str):
    """
    Converts the timeseries data into Gramian Angular Fields (GAFs) and maps them onto a image with 3 channels.

    Parameters
    ----------
    subject : str
        Subject ID.
    outpath : Path
        Path to save the GAFs.
    label : str
        Label of the subject (i.e., 'AD', 'FD', or 'Control').

    Returns
    -------
    im : np.ndarray
        GAFs of all trials of a subject.
    """
    path = Path(__file__).parents[1]
    ts_path  = path / 'data' / 'preprocessed' / f'{subject}_timeseries.npy'

    # loading in timeseries and labels
    X = np.load(ts_path)

    # loop over all trials
    for i in range(X.shape[0]):
        gaf = trial_to_gaf(X[i])

        # save each trial as a separate numpy array
        tmp_path = outpath / f'{subject}_{i}_{label}.npy'
        np.save(tmp_path, gaf)


def main():
    path = Path(__file__)
    
    image_size = 26

    data_path = path.parents[1] / 'data' / 'preprocessed'

    # load tsv file with diagnosis information
    tsv_path = path.parents[1] / 'data' / 'participants.tsv'
    df_diag = pd.read_csv(tsv_path, sep='\t', usecols=['participant_id', 'Group'])

    # loop over subjects
    subjects = [x for x in data_path.iterdir() if x.is_dir()]

    for subject in tqdm(subjects):
        outpath = path.parents[1] / 'data' / 'gaf' / subject.name

        # get label
        label = df_diag.loc[df_diag['participant_id'] == subject.name, 'Group'].iloc[0]
        # create directory
        if not outpath.exists():
            outpath.mkdir(parents=True)

        gaf_subject(subject, outpath, label=label)


if __name__ == '__main__':
    main()