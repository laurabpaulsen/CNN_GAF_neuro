import mne
from mne.datasets import sample
from pathlib import Path
import numpy as np

def get_data():
    data_path = sample.data_path()
    meg_path = data_path / 'MEG' / 'sample'
    raw_fname = meg_path / 'sample_audvis_raw.fif'

    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    events = mne.find_events(raw, stim_channel='STI 014')
    print(events.shape)

    return raw, events

def preprocess_meg(raw, events):
    picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=False, stim=False, exclude='bads')

    # filter raw data
    raw.filter(l_freq = 1, h_freq = 40)

    # epoch data
    epochs = mne.Epochs(raw, events, tmin=0, tmax=0.7, proj=True, picks=picks, baseline=None, preload=True)

    # resample epochs
    epochs.resample(250)

    return epochs



def main():
    path = Path(__file__)
    X_path = path.parent.parent / 'data' / 'timeseries_data'
    y_path = path.parent.parent / 'data' / 'timeseries_labels'
    raw, events = get_data()
    epochs = preprocess_meg(raw, events)

    # save epochs as numpy array
    X = epochs.get_data()
    y = epochs.events[:, -1]

    np.save(X_path, X)
    np.save(y_path, y)

    


if __name__ == '__main__':
    main()