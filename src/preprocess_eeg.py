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
    - Set the reference to common average
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
    
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False)

    # common average reference
    raw = raw.set_eeg_reference(ref_channels='average')

    # filter raw data
    raw.filter(l_freq = 1, h_freq = 40, verbose=False)

    # split into epochs of 10 seconds
    events = mne.make_fixed_length_events(raw, duration=5.0, overlap=0.0)

    # epoch data
    epochs = mne.Epochs(raw, events, tmin=0, tmax=5, reject={'eeg': 150e-6},  proj=True, picks=picks, baseline=None, preload=True, verbose=False)

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
    eeg_path = Path(f"data/raw/{subject}/eeg/{subject}_task-eyesclosed_eeg.set")
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
    outpath = path.parents[1] / 'data' / 'preprocessed'

    # loop over subjects
    subjects = [x.name for x in bids_path.iterdir() if x.is_dir()]

    for subject in tqdm(subjects):
        if subject.startswith('sub-'):

            # create directory
            if not outpath.exists():
                outpath.mkdir(parents = True)

            preprocess_subject(subject, outpath = outpath)


if __name__ == '__main__':
    main()