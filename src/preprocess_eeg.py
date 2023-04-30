"""
NOTES: Only one file we are interested in per subject.... maybe change preprocess subject????
"""

import mne
from mne.datasets import sample
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_data(eeg_path, event_path):
    raw = mne.io.read_raw_brainvision(eeg_path, preload=True, verbose=False)
    event_df =  pd.read_csv(event_path, sep = '\t', usecols=['stimulusnumber', 'onset', 'levelB'])

    # mapping between the second level label and the assigned event id
    event_id = return_eventids()

    events = []
    for index, row in event_df.iterrows():
        new_event = [row.onset, 0, event_id[row.levelB]]
        events.append(new_event)

    return raw, events

def return_eventids():
    return {'clothing': 0, 'fruits': 1, 'plants': 2, 'mammal': 3, 'human': 4, 'furniture': 5, 'aquatic':6, 'insect': 7, 'tools': 8, 'bird': 9, 'shapes': 10, 'object':11}

def preprocess_meg(raw, events):
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')

    # filter raw data
    raw.filter(l_freq = 1, h_freq = 40, verbose=False)

    # epoch data
    epochs = mne.Epochs(raw, events, tmin=0, tmax=0.2, proj=True, picks=picks, baseline=None, preload=True, verbose=False)

    # resample epochs
    epochs.resample(250)

    return epochs


def preprocess_subject(sub_path, out_path):

    # list all vhdr_files per in subject path
    p = sub_path.glob('**/*run-01_eeg.vhdr')
    vhdr_files = [x for x in p if x.is_file()]

    # get the tsv file with event information
    event_path = [str(x).split('_eeg.vhdr')[0]+'_events.tsv' for x in vhdr_files]

    for i, (eeg_path, event_path) in enumerate(zip(vhdr_files, event_path)):
        X_path = out_path / f'timeseries_run_{i+1}.npy'
        y_path = out_path / f'labels_run_{i+1}.npy'
        
        raw, events = get_data(eeg_path, event_path)

        epochs = preprocess_meg(raw, events)

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
    subject = [subject for subject in subjects if subject.name != "stimuli"]

    for subject in tqdm(subjects):
        out_path = path.parents[1] / 'data' / 'preprocessed' / subject.name

        # create directory
        if not out_path.exists():
            out_path.mkdir(parents = True)

        preprocess_subject(subject, out_path=out_path)


if __name__ == '__main__':
    main()