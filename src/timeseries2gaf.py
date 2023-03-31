"""
    Converts the timeseries data into Gramian Angular Fields (GAFs) and maps them onto a image with 3 channels.
    The GAFs are saved as numpy arrays in the data/gaf folder.
"""

import numpy as np
from pyts.image import GramianAngularField
from pathlib import Path
    

def main():
    path = Path(__file__)
    timeseries_path = path.parent.parent / 'data' / 'timeseries_data.npy'
    labels_path = path.parent.parent / 'data' / 'timeseries_labels.npy'

    X = np.load(timeseries_path)
    y = np.load(labels_path)

    image_size = 38

    transformer = GramianAngularField(image_size)
    gaf_path = path.parent.parent / 'data' / 'gaf'
    
    # loop over all trials
    for i in range(X.shape[0]):

        # transform each trial into a GAF
        X_gaf = transformer.fit_transform(X[i])
        X_gaf = np.stack((X_gaf, X_gaf, X_gaf), axis=2)

        # save each trial as a separate numpy array
        tmp_path = gaf_path / f'trial_{i}_label_{y[i]}.npy'
        np.save(tmp_path, X_gaf)

if __name__ == '__main__':
    main()