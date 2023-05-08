"""

"""

import mne
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

def get_data(eeg_path, event_path):
    raw = mne.io.read_raw_brainvision(eeg_path, preload=True, verbose=False)
    event_df =  pd.read_csv(event_path, sep = '\t', usecols=['stimulusnumber', 'onset', 'levelA'])
    event_df = event_df[event_df["levelA"] != 'targets']

    # mapping between the second level label and the assigned event id
    event_id = return_eventids()

    events = []
    for _, row in event_df.iterrows():
        #print(row.levelA)
        new_event = [row.onset, 0, event_id[row.levelA]]
        events.append(new_event)

    return raw, events

def return_eventids():
    return {"animate": 0, "inanimate": 1}

def preprocess_eeg(raw, events):
    """
    Performs the following preprocessing steps on the raw EEG data.
    - Set the reference to common average
    - Filters the data between 1 and 40 Hz.
    - Splits the data into epochs of 10 seconds.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    events : list
        List of events.
    
    Returns
    -------
    epochs : mne.Epochs
        Preprocessed epoched EEG data. 
    """
    
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False)

    # common average reference
    raw.set_eeg_reference('average', projection=True, verbose=False)

    # filter raw data
    raw.filter(l_freq = 1, h_freq = 40, verbose=False)

    # epoch data
    epochs = mne.Epochs(raw, events, tmin=0, tmax=0.5, proj=True, picks=picks, baseline=None, preload=True, verbose=False, reject ={'eeg': 150e-6})
    return epochs

def preprocess_subject(sub_path:Path):
    path = Path(__file__)
    out_path = path.parents[1] / 'data' / 'preprocessed' / sub_path.name

    # create output directory if it does not exist
    if not out_path.exists():
        out_path.mkdir(parents=True)

    # get path for the run_01_eeg.vhdr file
    vhdr_path = sub_path / "eeg" / f"{sub_path.name}_task-rsvp_run-01_eeg.vhdr"

    # tsv file with event information
    event_path = sub_path / "eeg" / f'{sub_path.name}_task-rsvp_run-01_events.tsv'

    X_path = out_path / f'X.npy'
    y_path = out_path / f'y.npy'
        
    raw, events = get_data(vhdr_path, event_path)

    epochs = preprocess_eeg(raw, events)

    # save epochs as numpy array
    X = epochs.get_data()
    y = epochs.events[:, -1]

    np.save(X_path, X)
    np.save(y_path, y)


def main():
    path = Path(__file__)

    bids_path = path.parents[1] / 'data' / 'raw'

    # loop over subjects
    subjects = [x for x in bids_path.iterdir() if x.is_dir()]
    subjects = [subject for subject in subjects if subject.name.startswith("sub-")]

    # use multiprocessing to speed up the process
    pool = mp.Pool(mp.cpu_count())
    pool.map(preprocess_subject, subjects)


if __name__ == '__main__':
    main()