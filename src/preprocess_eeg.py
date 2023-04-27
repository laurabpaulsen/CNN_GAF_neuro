"""
NOTES: Only one file we are interested in per subject.... maybe change preprocess subject????
"""

import mne
from pathlib import Path
import numpy as np
from tqdm import tqdm


def get_data(eeg_path):
    # read .set files
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose=False)

    return raw

def preprocess_eeg(raw):
    """
    Performs the following preprocessing steps on the raw EEG data.
    - Filters the data between 1 and 40 Hz.
    - Splits the data into epochs of 10 seconds.
    - Resamples the data to 200 Hz.


    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    
    Returns
    -------
    epochs : mne.Epochs
        Preprocessed epoched EEG data. 
    """
    
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')

    # filter raw data
    raw.filter(l_freq = 1, h_freq = 40, verbose=False)

    # split into epochs of 10 seconds
    events = mne.make_fixed_length_events(raw, duration=10.0, overlap=0.0, verbose=False)

    # epoch data
    epochs = mne.Epochs(raw, events, tmin=0, tmax=10, proj=True, picks=picks, baseline=None, preload=True, verbose=False)

    # resample epochs
    epochs.resample(200)

    return epochs


def preprocess_subject(subject:str, outpath:Path):
    """
    Preprocesses the data of a single subject and saves it as a numpy array.

    Parameters
    ----------
    subject : str
        Subject ID.
    outpath : Path
        Path to save the preprocessed data.
    """
    eeg_path = Path(f"data/{subject}/eeg/{subject}_task-eyesclosed_eeg.set")
    X_path = outpath / f'{subject}_timeseries.npy'

    # get data
    raw = get_data(eeg_path)
    epochs = preprocess_eeg(raw)

    # save epochs as numpy array
    X = epochs.get_data()
    np.save(X_path, X)


def main():
    path = Path(__file__)

    bids_path = path.parents[1] / 'data' / 'raw'

    # loop over subjects
    subjects = [x for x in bids_path.iterdir() if x.is_dir()]

    for subject in tqdm(subjects):
        if subject.name.startswith('sub-'):
            outpath = path.parents[1] / 'data' / 'preprocessed'

            # create directory
            if not outpath.exists():
                outpath.mkdir(parents = True)

            preprocess_subject(subject.name, outpath = outpath)


if __name__ == '__main__':
    main()